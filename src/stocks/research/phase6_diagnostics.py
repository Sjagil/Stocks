from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file, utc_now_iso
from stocks.research.phase6 import (
    Phase6Layout,
    _annualized_vol,
    _cagr,
    _cash_key,
    _equal_weight_targets,
    _inverse_vol_targets,
    _max_drawdown,
    _mean,
    _median,
    _momentum_targets,
    _portfolio_result,
    _prepare_common_history,
    _profit_factor,
    _slice_prepared,
    _std,
    _trend_targets,
    _write_json,
    _zero_financial_calls,
    load_phase6_dataset,
)


PHASE6_1_DECISIONS = (
    "NO_FINANCIAL_FINALIST",
    "PROMISING_RESEARCH_CANDIDATE",
    "FINANCIAL_FINALIST_GO",
    "METRIC_IMPLEMENTATION_BLOCKED",
    "INSUFFICIENT_SAMPLE",
)


@dataclass(frozen=True)
class Phase61Layout:
    output_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> Phase61Layout:
        return cls(output_dir=project_root / "output" / "research" / "phase6_1")

    @property
    def fold_diagnostics_json(self) -> Path:
        return self.output_dir / "fold-diagnostics.json"

    @property
    def benchmark_comparison_json(self) -> Path:
        return self.output_dir / "benchmark-comparison.json"

    @property
    def concentration_json(self) -> Path:
        return self.output_dir / "concentration.json"

    @property
    def leave_one_out_json(self) -> Path:
        return self.output_dir / "leave-one-out.json"

    @property
    def regime_analysis_json(self) -> Path:
        return self.output_dir / "regime-analysis.json"

    @property
    def cost_stress_json(self) -> Path:
        return self.output_dir / "cost-stress.json"

    @property
    def parameter_plateau_json(self) -> Path:
        return self.output_dir / "parameter-plateau.json"

    @property
    def multiple_testing_json(self) -> Path:
        return self.output_dir / "multiple-testing.json"

    @property
    def monte_carlo_json(self) -> Path:
        return self.output_dir / "bootstrap-monte-carlo.json"

    @property
    def sample_size_json(self) -> Path:
        return self.output_dir / "sample-size-gate.json"

    @property
    def status_json(self) -> Path:
        return self.output_dir / "phase6_1-status.json"

    @property
    def freeze_json(self) -> Path:
        return self.output_dir / "phase6_1-freeze.json"


def phase6_1_schema() -> dict[str, Any]:
    return {
        "schema": "phase6_1_robustness_failure_attribution_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "decision_statuses": list(PHASE6_1_DECISIONS),
        "diagnostics": [
            "walk_forward_fold_diagnostics",
            "profit_factor_zero_reason",
            "benchmark_excess_per_fold",
            "asset_region_sleeve_concentration",
            "leave_one_out",
            "regime_analysis",
            "cost_stress",
            "parameter_plateau",
            "multiple_testing",
            "block_bootstrap_monte_carlo",
            "sample_size_gate",
        ],
        "authority": {
            "optimizer_enabled": False,
            "paper_orders_enabled": False,
            "provider_calls_enabled": False,
            "broker_calls_enabled": False,
        },
        "financial_calls": _zero_financial_calls(),
    }


def run_phase6_1_pipeline(project_root: Path) -> dict[str, Any]:
    layout = Phase61Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_phase6_dataset(project_root)
    prepared = _prepare_common_history(dataset)
    phase6_grid = _load_phase6_grid(project_root)
    canonical_config = _canonical_config(phase6_grid)

    folds = fold_diagnostics(prepared)
    _write_json(layout.fold_diagnostics_json, folds)
    benchmark_comparison = benchmark_comparisons(prepared, folds)
    _write_json(layout.benchmark_comparison_json, benchmark_comparison)
    concentration = concentration_report(prepared, canonical_config)
    _write_json(layout.concentration_json, concentration)
    loo = leave_one_out_report(prepared, canonical_config)
    _write_json(layout.leave_one_out_json, loo)
    regimes = regime_analysis(prepared, dataset, canonical_config)
    _write_json(layout.regime_analysis_json, regimes)
    costs = cost_stress_report(prepared, canonical_config)
    _write_json(layout.cost_stress_json, costs)
    plateau = parameter_plateau_report(phase6_grid)
    _write_json(layout.parameter_plateau_json, plateau)
    multiple = multiple_testing_report(prepared, folds, phase6_grid)
    _write_json(layout.multiple_testing_json, multiple)
    monte_carlo = monte_carlo_report(prepared, canonical_config)
    _write_json(layout.monte_carlo_json, monte_carlo)
    sample = sample_size_gate(folds)
    _write_json(layout.sample_size_json, sample)
    status = phase6_1_status(
        project_root,
        folds=folds,
        benchmark_comparison=benchmark_comparison,
        concentration=concentration,
        leave_one_out=loo,
        regimes=regimes,
        costs=costs,
        plateau=plateau,
        multiple=multiple,
        monte_carlo=monte_carlo,
        sample=sample,
    )
    _write_json(layout.status_json, status)
    return status


def phase6_1_status(
    project_root: Path,
    *,
    folds: dict[str, Any] | None = None,
    benchmark_comparison: dict[str, Any] | None = None,
    concentration: dict[str, Any] | None = None,
    leave_one_out: dict[str, Any] | None = None,
    regimes: dict[str, Any] | None = None,
    costs: dict[str, Any] | None = None,
    plateau: dict[str, Any] | None = None,
    multiple: dict[str, Any] | None = None,
    monte_carlo: dict[str, Any] | None = None,
    sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layout = Phase61Layout.from_project_root(project_root)
    folds = folds or _read_json_if_exists(layout.fold_diagnostics_json)
    benchmark_comparison = benchmark_comparison or _read_json_if_exists(layout.benchmark_comparison_json)
    concentration = concentration or _read_json_if_exists(layout.concentration_json)
    leave_one_out = leave_one_out or _read_json_if_exists(layout.leave_one_out_json)
    regimes = regimes or _read_json_if_exists(layout.regime_analysis_json)
    costs = costs or _read_json_if_exists(layout.cost_stress_json)
    plateau = plateau or _read_json_if_exists(layout.parameter_plateau_json)
    multiple = multiple or _read_json_if_exists(layout.multiple_testing_json)
    monte_carlo = monte_carlo or _read_json_if_exists(layout.monte_carlo_json)
    sample = sample or _read_json_if_exists(layout.sample_size_json)
    checks = {
        "fold_diagnostics_available": folds is not None and folds.get("fold_count", 0) > 0,
        "pf_zero_reasons_available": folds is not None and all(fold.get("PF_zero_reason") is not None for fold in folds.get("folds", []) if fold.get("period_profit_factor") == 0 or fold.get("episode_profit_factor") == 0),
        "benchmark_comparisons_available": benchmark_comparison is not None and benchmark_comparison.get("comparison_count", 0) > 0,
        "concentration_available": concentration is not None and concentration.get("status") == "GO",
        "leave_one_out_available": leave_one_out is not None and leave_one_out.get("status") == "GO",
        "regime_analysis_available": regimes is not None and regimes.get("status") == "GO",
        "cost_stress_available": costs is not None and costs.get("status") == "GO",
        "parameter_plateau_available": plateau is not None and plateau.get("status") == "GO",
        "multiple_testing_available": multiple is not None and multiple.get("status") == "GO",
        "monte_carlo_available": monte_carlo is not None and monte_carlo.get("status") == "GO",
        "sample_size_gate_available": sample is not None and sample.get("status") in {"GO", "INSUFFICIENT_SAMPLE"},
        "no_financial_calls": True,
    }
    decision = _decision_status(folds, benchmark_comparison, concentration, costs, plateau, multiple, monte_carlo, sample)
    return {
        "schema": "phase6_1_robustness_failure_attribution_status_v1",
        "status": "PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_GO" if all(checks.values()) else "NO_GO",
        "decision_status": decision,
        "generated_at": utc_now_iso(),
        "checks": checks,
        "summary": {
            "fold_count": None if folds is None else folds.get("fold_count"),
            "positive_folds": None if folds is None else folds.get("positive_folds"),
            "worst_period_pf": None if folds is None else folds.get("worst_period_pf"),
            "worst_pf_zero_reason": None if folds is None else folds.get("worst_pf_zero_reason"),
            "benchmark_excess_pass_count": None if benchmark_comparison is None else benchmark_comparison.get("strategy_beats_benchmark_count"),
            "dominance_flags": None if concentration is None else concentration.get("dominance_flags"),
            "break_even_cost_bps": None if costs is None else costs.get("break_even_cost_bps"),
            "sample_gate_status": None if sample is None else sample.get("status"),
            "pbo": None if multiple is None else multiple.get("probability_of_backtest_overfitting"),
            "dsr_probability": None if multiple is None else multiple.get("deflated_sharpe_probability"),
        },
        "artifacts": {
            "fold_diagnostics": str(layout.fold_diagnostics_json),
            "benchmark_comparison": str(layout.benchmark_comparison_json),
            "concentration": str(layout.concentration_json),
            "leave_one_out": str(layout.leave_one_out_json),
            "regime_analysis": str(layout.regime_analysis_json),
            "cost_stress": str(layout.cost_stress_json),
            "parameter_plateau": str(layout.parameter_plateau_json),
            "multiple_testing": str(layout.multiple_testing_json),
            "bootstrap_monte_carlo": str(layout.monte_carlo_json),
            "sample_size_gate": str(layout.sample_size_json),
        },
        "financial_calls": _zero_financial_calls(),
    }


def fold_diagnostics(prepared: dict[str, Any]) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    dates = prepared["dates"]
    start_year = date.fromisoformat(dates[0]).year
    end_year = date.fromisoformat(dates[-1]).year
    candidates = _config_candidates()
    for fold_id, train_start_year in enumerate(range(start_year, end_year - 4), start=1):
        train_start = f"{train_start_year}-01-01"
        train_end = f"{train_start_year + 2}-12-31"
        validation_start = f"{train_start_year + 3}-01-01"
        validation_end = f"{train_start_year + 3}-12-31"
        test_start = f"{train_start_year + 4}-01-01"
        test_end = f"{train_start_year + 4}-12-31"
        if test_end > dates[-1]:
            break
        validation_subset = _slice_prepared(prepared, validation_start, validation_end)
        validation_scores: list[tuple[float, dict[str, Any]]] = []
        for config in candidates:
            result = _portfolio_result(
                "phase6_1_validation_selection",
                validation_subset,
                _targets_for_config(validation_subset, config),
                cost_bps=float(config["cost_bps"]),
                parameters=config,
            )
            validation_scores.append((float(result["net_return"]), config))
        selected = max(validation_scores, key=lambda item: item[0])[1]
        test_subset = _slice_prepared(prepared, test_start, test_end)
        detailed = _detailed_portfolio(
            test_subset,
            _targets_for_config(test_subset, selected),
            cost_bps=float(selected["cost_bps"]),
            parameters=selected,
            name="walk_forward_test",
        )
        sample_status = _sample_status(
            detailed["metrics"]["closed_position_episodes"],
            len({row["date"][:7] for row in detailed["daily"]}),
            detailed["metrics"]["rebalance_count"],
            len(detailed["daily"]),
        )
        fold = {
            "fold_id": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "test_start": test_start,
            "test_end": test_end,
            "selected_configuration": selected,
            "eligible_instruments": detailed["eligible_instruments"],
            "trade_episodes": detailed["metrics"]["closed_position_episodes"],
            "positive_episodes": detailed["episode_summary"]["positive_episodes"],
            "negative_episodes": detailed["episode_summary"]["negative_episodes"],
            "zero_pnl_episodes": detailed["episode_summary"]["zero_pnl_episodes"],
            "gross_pnl": detailed["gross_pnl"],
            "net_pnl": detailed["net_pnl"],
            "period_profit_factor": detailed["metrics"]["period_profit_factor"],
            "episode_profit_factor": detailed["metrics"]["trade_profit_factor"],
            "PF_zero_reason": _pf_zero_reason(detailed),
            "period_pf_diagnostics": _pf_diagnostics([row["net_return"] for row in detailed["daily"]], sample_name="period"),
            "episode_pf_diagnostics": _pf_diagnostics([episode["pnl"] for episode in detailed["episodes"]], sample_name="episode"),
            "CAGR": detailed["metrics"]["CAGR"],
            "Sharpe": detailed["metrics"]["Sharpe"],
            "maximum_drawdown": detailed["metrics"]["maximum_drawdown"],
            "turnover": detailed["metrics"]["turnover"],
            "rebalance_count": detailed["metrics"]["rebalance_count"],
            "costs": detailed["cost_amount"],
            "average_cash": detailed["metrics"]["average_cash"],
            "region_exposure": detailed["metrics"]["region_exposure"],
            "sleeve_exposure": detailed["metrics"]["sleeve_exposure"],
            "sample_status": sample_status,
            "active_months": len({row["date"][:7] for row in detailed["daily"]}),
            "daily_returns": [row["net_return"] for row in detailed["daily"]],
        }
        folds.append(fold)
    return {
        "schema": "phase6_1_fold_diagnostics_v1",
        "status": "GO" if folds else "NO_GO",
        "fold_count": len(folds),
        "positive_folds": sum(1 for fold in folds if fold["net_pnl"] > 0),
        "worst_period_pf": min((fold["period_profit_factor"] for fold in folds if fold["period_profit_factor"] is not None), default=None),
        "worst_pf_zero_reason": next((fold["PF_zero_reason"] for fold in folds if fold["period_profit_factor"] == 0 or fold["episode_profit_factor"] == 0), None),
        "folds": folds,
        "financial_calls": _zero_financial_calls(),
    }


def benchmark_comparisons(prepared: dict[str, Any], folds: dict[str, Any]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    benchmark_builders = {
        "equal_weight": lambda subset: _equal_weight_targets(subset, "monthly"),
        "inverse_volatility": lambda subset: _inverse_vol_targets(subset, lookback=63, rebalance="monthly"),
        "worldindex_buy_and_hold": lambda subset: [{_world_index_key(subset): 1.0} for _ in subset["dates"]],
        "trend_200": lambda subset: _trend_targets(subset, trend_lookback=200, rebalance="monthly"),
        "simple_momentum_rotation": lambda subset: _momentum_targets(subset, momentum_lookback=252, trend_lookback=200, top_n=4, rebalance="monthly"),
    }
    for fold in folds["folds"]:
        subset = _slice_prepared(prepared, fold["test_start"], fold["test_end"])
        strategy = _detailed_portfolio(
            subset,
            _targets_for_config(subset, fold["selected_configuration"]),
            cost_bps=float(fold["selected_configuration"]["cost_bps"]),
            parameters=fold["selected_configuration"],
            name="strategy",
        )
        strategy_returns = [row["net_return"] for row in strategy["daily"]]
        for benchmark_name, builder in benchmark_builders.items():
            benchmark = _detailed_portfolio(subset, builder(subset), cost_bps=10.0, parameters={}, name=benchmark_name)
            benchmark_returns = [row["net_return"] for row in benchmark["daily"]]
            excess = [left - right for left, right in zip(strategy_returns, benchmark_returns, strict=False)]
            comparisons.append(
                {
                    "fold_id": fold["fold_id"],
                    "benchmark": benchmark_name,
                    "strategy_net_return": strategy["metrics"]["net_return"],
                    "benchmark_net_return": benchmark["metrics"]["net_return"],
                    "excess_CAGR": _excess_cagr(excess, subset["dates"]),
                    "excess_Sharpe": _sharpe(excess),
                    "drawdown_improvement": strategy["metrics"]["maximum_drawdown"] - benchmark["metrics"]["maximum_drawdown"],
                    "turnover_difference": strategy["metrics"]["turnover"] - benchmark["metrics"]["turnover"],
                    "cost_difference": strategy["cost_amount"] - benchmark["cost_amount"],
                    "tracking_error": _annualized_vol(excess),
                    "information_ratio": _information_ratio(excess),
                }
            )
    return {
        "schema": "phase6_1_benchmark_comparison_v1",
        "status": "GO",
        "comparison_count": len(comparisons),
        "strategy_beats_benchmark_count": sum(1 for item in comparisons if item["excess_CAGR"] is not None and item["excess_CAGR"] > 0),
        "comparisons": comparisons,
        "financial_calls": _zero_financial_calls(),
    }


def concentration_report(prepared: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    detailed = _detailed_portfolio(prepared, _targets_for_config(prepared, config), cost_bps=float(config["cost_bps"]), parameters=config, name="canonical")
    positive_asset = _positive_share(detailed["contribution_by_asset"])
    region_contribution = _group_contribution(detailed["contribution_by_asset"], prepared["metadata"], "region")
    sleeve_contribution = _group_contribution(detailed["contribution_by_asset"], prepared["metadata"], "sleeve")
    positive_region = _positive_share(region_contribution)
    positive_sleeve = _positive_share(sleeve_contribution)
    positive_year = _positive_share(detailed["contribution_by_year"])
    flags = {
        "single_asset_gt_40pct": max(positive_asset.values(), default=0.0) > 0.40,
        "single_region_gt_60pct": max(positive_region.values(), default=0.0) > 0.60,
        "single_sleeve_gt_70pct": max(positive_sleeve.values(), default=0.0) > 0.70,
        "single_year_gt_50pct": max(positive_year.values(), default=0.0) > 0.50,
    }
    return {
        "schema": "phase6_1_concentration_v1",
        "status": "GO",
        "configuration": config,
        "asset_contribution": _symbolized(detailed["contribution_by_asset"], prepared["metadata"]),
        "asset_positive_contribution_share": _symbolized(positive_asset, prepared["metadata"]),
        "region_contribution": region_contribution,
        "region_positive_contribution_share": positive_region,
        "sleeve_contribution": sleeve_contribution,
        "sleeve_positive_contribution_share": positive_sleeve,
        "year_contribution": detailed["contribution_by_year"],
        "year_positive_contribution_share": positive_year,
        "dominance_flags": flags,
        "financial_calls": _zero_financial_calls(),
    }


def leave_one_out_report(prepared: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    full = _detailed_portfolio(prepared, _targets_for_config(prepared, config), cost_bps=float(config["cost_bps"]), parameters=config, name="full")
    results: list[dict[str, Any]] = []
    removals: list[tuple[str, str, set[str]]] = []
    for key, meta in prepared["metadata"].items():
        removals.append(("instrument", str(meta["symbol"]), {key}))
    for region in sorted({str(meta["region"]) for meta in prepared["metadata"].values()}):
        removals.append(("region", region, {key for key, meta in prepared["metadata"].items() if meta["region"] == region}))
    for sleeve in sorted({str(meta["sleeve"]) for meta in prepared["metadata"].values()}):
        removals.append(("sleeve", sleeve, {key for key, meta in prepared["metadata"].items() if meta["sleeve"] == sleeve}))
    seen: set[tuple[str, str]] = set()
    for removal_type, removal_id, keys in removals:
        if (removal_type, removal_id) in seen or len(keys) >= len(prepared["returns"]):
            continue
        seen.add((removal_type, removal_id))
        subset = _remove_keys(prepared, keys)
        adjusted_config = {**config}
        result = _detailed_portfolio(subset, _targets_for_config(subset, adjusted_config), cost_bps=float(config["cost_bps"]), parameters=adjusted_config, name="leave_one_out")
        results.append(
            {
                "removal_type": removal_type,
                "removal_id": removal_id,
                "removed_count": len(keys),
                "net_return": result["metrics"]["net_return"],
                "period_profit_factor": result["metrics"]["period_profit_factor"],
                "Sharpe": result["metrics"]["Sharpe"],
                "maximum_drawdown": result["metrics"]["maximum_drawdown"],
                "sensitivity": {
                    "net_return_delta": full["metrics"]["net_return"] - result["metrics"]["net_return"],
                    "period_profit_factor_delta": _nullable_delta(full["metrics"]["period_profit_factor"], result["metrics"]["period_profit_factor"]),
                    "Sharpe_delta": _nullable_delta(full["metrics"]["Sharpe"], result["metrics"]["Sharpe"]),
                    "maximum_drawdown_delta": full["metrics"]["maximum_drawdown"] - result["metrics"]["maximum_drawdown"],
                },
            }
        )
    return {
        "schema": "phase6_1_leave_one_out_v1",
        "status": "GO",
        "full_configuration": config,
        "full_net_return": full["metrics"]["net_return"],
        "test_count": len(results),
        "fragility_flags": {
            "any_net_return_below_zero": any(item["net_return"] < 0 for item in results),
            "any_pf_below_one": any((item["period_profit_factor"] or 0) < 1 for item in results),
        },
        "results": results,
        "financial_calls": _zero_financial_calls(),
    }


def regime_analysis(prepared: dict[str, Any], dataset: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    detailed = _detailed_portfolio(prepared, _targets_for_config(prepared, config), cost_bps=float(config["cost_bps"]), parameters=config, name="canonical")
    daily_returns = [row["net_return"] for row in detailed["daily"]]
    regime_by_date = _regime_labels(prepared, dataset)
    results: list[dict[str, Any]] = []
    for regime in sorted({label for labels in regime_by_date.values() for label in labels}):
        selected = [index for index, row in enumerate(detailed["daily"]) if regime in regime_by_date.get(row["date"], set())]
        returns = [daily_returns[index] for index in selected]
        turnovers = [detailed["daily"][index]["turnover"] for index in selected]
        cash = [detailed["daily"][index]["cash_weight"] for index in selected]
        regime_dates = [detailed["daily"][index]["date"] for index in selected]
        episode_count = sum(1 for episode in detailed["episodes"] if regime in regime_by_date.get(episode["end_date"], set()))
        results.append(
            {
                "regime": regime,
                "observations": len(returns),
                "episodes": episode_count,
                "net_return": _compound_return(returns),
                "PF": _profit_factor(returns),
                "Sharpe": _sharpe(returns),
                "maximum_drawdown": _max_drawdown(_nav_path(returns)),
                "turnover": sum(turnovers),
                "cash_exposure": _mean(cash),
                "first_date": regime_dates[0] if regime_dates else None,
                "last_date": regime_dates[-1] if regime_dates else None,
            }
        )
    return {
        "schema": "phase6_1_regime_analysis_v1",
        "status": "GO",
        "results": results,
        "financial_calls": _zero_financial_calls(),
    }


def cost_stress_report(prepared: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cost_bps in (5.0, 10.0, 20.0, 30.0, 50.0):
        stressed_config = {**config, "cost_bps": cost_bps}
        detailed = _detailed_portfolio(prepared, _targets_for_config(prepared, stressed_config), cost_bps=cost_bps, parameters=stressed_config, name="cost_stress")
        rows.append(
            {
                "cost_bps": cost_bps,
                "net_return": detailed["metrics"]["net_return"],
                "period_profit_factor": detailed["metrics"]["period_profit_factor"],
                "Sharpe": detailed["metrics"]["Sharpe"],
                "maximum_drawdown": detailed["metrics"]["maximum_drawdown"],
                "turnover": detailed["metrics"]["turnover"],
                "cost_amount": detailed["cost_amount"],
            }
        )
    base = _detailed_portfolio(prepared, _targets_for_config(prepared, config), cost_bps=0.0, parameters=config, name="break_even")
    break_even = None if base["turnover_amount"] <= 0 else base["gross_pnl"] / base["turnover_amount"] * 10_000.0
    return {
        "schema": "phase6_1_cost_stress_v1",
        "status": "GO",
        "configuration": config,
        "break_even_cost_bps": break_even,
        "results": rows,
        "financial_calls": _zero_financial_calls(),
    }


def parameter_plateau_report(grid: dict[str, Any]) -> dict[str, Any]:
    results = grid.get("results", [])
    dimensions = {
        "momentum_lookback": (63, 126, 252),
        "trend_lookback": (100, 200),
        "rebalance": ("weekly", "monthly"),
        "target_vol": (0.08, 0.10, 0.12),
        "cost_bps": (5.0, 10.0, 20.0),
    }
    rows: dict[str, dict[str, Any]] = {}
    for dimension, values in dimensions.items():
        rows[dimension] = {}
        for value in values:
            bucket = [item for item in results if item.get("parameters", {}).get(dimension) == value]
            rows[dimension][str(value)] = {
                "config_count": len(bucket),
                "median_PF": _median([item["period_profit_factor"] for item in bucket if item["period_profit_factor"] is not None]),
                "median_Sharpe": _median([item["Sharpe"] for item in bucket if item["Sharpe"] is not None]),
                "positive_fold_ratio": sum(1 for item in bucket if item["net_return"] > 0 and (item["period_profit_factor"] or 0) > 1.0) / len(bucket) if bucket else None,
                "median_drawdown": _median([item["maximum_drawdown"] for item in bucket]),
            }
    return {
        "schema": "phase6_1_parameter_plateau_v1",
        "status": "GO" if results else "NO_GO",
        "dimensions": rows,
        "has_plateau": grid.get("plateau", {}).get("has_plateau") is True,
        "financial_calls": _zero_financial_calls(),
    }


def multiple_testing_report(prepared: dict[str, Any], folds: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    results = grid.get("results", [])
    sharpes = [item["Sharpe"] for item in results if item.get("Sharpe") is not None]
    best_sharpe = max(sharpes) if sharpes else None
    mean_sharpe = _mean(sharpes)
    std_sharpe = _std(sharpes)
    deflated_probability = None if best_sharpe is None or std_sharpe == 0 else _normal_cdf((best_sharpe - mean_sharpe) / std_sharpe - math.sqrt(2.0 * math.log(max(2, len(sharpes)))))
    pbo_flags = []
    for fold in folds.get("folds", []):
        subset = _slice_prepared(prepared, fold["test_start"], fold["test_end"])
        test_scores = []
        for config in _config_candidates():
            result = _portfolio_result(
                "phase6_1_pbo_test",
                subset,
                _targets_for_config(subset, config),
                cost_bps=float(config["cost_bps"]),
                parameters=config,
            )
            test_scores.append(result["net_return"])
        selected_result = _portfolio_result(
            "phase6_1_pbo_selected",
            subset,
            _targets_for_config(subset, fold["selected_configuration"]),
            cost_bps=float(fold["selected_configuration"]["cost_bps"]),
            parameters=fold["selected_configuration"],
        )
        median_score = _median(test_scores)
        pbo_flags.append(median_score is not None and selected_result["net_return"] < median_score)
    pbo = sum(1 for flag in pbo_flags if flag) / len(pbo_flags) if pbo_flags else None
    return {
        "schema": "phase6_1_multiple_testing_v1",
        "status": "GO",
        "tested_configurations": len(results),
        "best_raw_sharpe": best_sharpe,
        "mean_config_sharpe": mean_sharpe,
        "std_config_sharpe": std_sharpe,
        "deflated_sharpe_probability": deflated_probability,
        "probability_of_backtest_overfitting": pbo,
        "oos_rank_stability": "ACCEPTABLE" if pbo is not None and pbo < 0.5 else "WEAK",
        "bootstrap_confidence_source": "phase6_1_block_bootstrap_monte_carlo",
        "financial_calls": _zero_financial_calls(),
    }


def monte_carlo_report(prepared: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    detailed = _detailed_portfolio(prepared, _targets_for_config(prepared, config), cost_bps=float(config["cost_bps"]), parameters=config, name="canonical")
    returns = [row["net_return"] for row in detailed["daily"]]
    episodes = [episode["pnl"] for episode in detailed["episodes"]]
    rng = random.Random(610_001)
    terminal_navs: list[float] = []
    drawdowns: list[float] = []
    pfs: list[float | None] = []
    iterations = 300
    for _ in range(iterations):
        path = _block_bootstrap(returns, block_size=21, rng=rng)
        navs = _nav_path(path)
        terminal_navs.append(navs[-1])
        drawdowns.append(abs(_max_drawdown(navs)))
        pfs.append(_profit_factor(path))
    episode_pfs = []
    for _ in range(iterations):
        if episodes:
            sample = [episodes[rng.randrange(len(episodes))] for _ in episodes]
            episode_pfs.append(_profit_factor(sample))
    return {
        "schema": "phase6_1_bootstrap_monte_carlo_v1",
        "status": "GO",
        "method": {
            "block_bootstrap_daily_returns": {"iterations": iterations, "block_size": 21},
            "trade_episode_bootstrap": {"iterations": iterations},
            "cost_perturbation": "covered_by_cost_stress_5_10_20_30_50_bps",
            "start_date_perturbation": _start_date_perturbation(returns),
            "rebalance_date_perturbation": "approximated_by_weekly_vs_monthly_grid_dimension",
        },
        "median_terminal_NAV": _median(terminal_navs),
        "P5_terminal_NAV": _percentile(terminal_navs, 5),
        "P95_terminal_NAV": _percentile(terminal_navs, 95),
        "probability_of_loss": sum(1 for value in terminal_navs if value < 100.0) / len(terminal_navs),
        "median_maximum_drawdown": _median(drawdowns),
        "P95_maximum_drawdown": _percentile(drawdowns, 95),
        "probability_PF_gt_1": sum(1 for value in pfs if value is not None and value > 1.0) / len(pfs),
        "episode_bootstrap_probability_PF_gt_1": sum(1 for value in episode_pfs if value is not None and value > 1.0) / len(episode_pfs) if episode_pfs else None,
        "financial_calls": _zero_financial_calls(),
    }


def sample_size_gate(folds: dict[str, Any]) -> dict[str, Any]:
    rows = []
    statuses = []
    for fold in folds.get("folds", []):
        active_months = int(fold.get("active_months", 0))
        status = _sample_status(fold["trade_episodes"], active_months, fold["rebalance_count"] if "rebalance_count" in fold else 0, len(fold.get("daily_returns", [])))
        statuses.append(status)
        rows.append(
            {
                "fold_id": fold["fold_id"],
                "closed_episodes": fold["trade_episodes"],
                "active_months": active_months,
                "rebalances": fold.get("rebalance_count", fold.get("turnover", 0)),
                "effective_sample_size": _effective_sample_size(fold.get("daily_returns", [])),
                "sample_status": status,
            }
        )
    aggregate_status = "INSUFFICIENT_SAMPLE" if any(row["sample_status"] == "INSUFFICIENT_SAMPLE" for row in rows) else "GO"
    return {
        "schema": "phase6_1_sample_size_gate_v1",
        "status": aggregate_status,
        "folds": rows,
        "thresholds": {"insufficient_episodes_lt": 10, "low_confidence_episodes_lt": 30},
        "financial_calls": _zero_financial_calls(),
    }


def phase6_1_freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase61Layout.from_project_root(project_root)
    status = phase6_1_status(project_root)
    source_paths = (
        "main.py",
        "src/stocks/research/phase6.py",
        "src/stocks/research/phase6_diagnostics.py",
        "tests/test_phase6_research.py",
        "tests/test_phase6_diagnostics.py",
    )
    artifact_paths = (
        layout.fold_diagnostics_json,
        layout.benchmark_comparison_json,
        layout.concentration_json,
        layout.leave_one_out_json,
        layout.regime_analysis_json,
        layout.cost_stress_json,
        layout.parameter_plateau_json,
        layout.multiple_testing_json,
        layout.monte_carlo_json,
        layout.sample_size_json,
        layout.status_json,
    )
    freeze = {
        "schema": "phase6_1_freeze_v1",
        "contract_id": "PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_V1",
        "freeze_status": "PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_FROZEN_GO"
        if status["status"] == "PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_GO"
        else "NO_GO",
        "phase6_1_status": status["status"],
        "decision_status": status["decision_status"],
        "generated_at": utc_now_iso(),
        "source_hashes": {path: sha256_file(project_root / path) for path in source_paths},
        "artifact_hashes": {str(path): sha256_file(path) for path in artifact_paths},
        "financial_calls": _zero_financial_calls(),
    }
    _write_json(layout.freeze_json, freeze)
    return freeze


def _detailed_portfolio(
    prepared: dict[str, Any],
    weights_by_period: list[dict[str, float]],
    *,
    cost_bps: float,
    parameters: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    dates = prepared["dates"]
    returns = prepared["returns"]
    metadata = prepared["metadata"]
    nav = 100.0
    previous_weights: dict[str, float] = {"CASH": 1.0}
    daily: list[dict[str, Any]] = []
    navs = [nav]
    net_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    cash_weights: list[float] = []
    active: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    contribution_by_asset: dict[str, float] = {}
    contribution_by_year: dict[str, float] = {}
    gross_pnl = 0.0
    cost_amount = 0.0
    turnover_amount = 0.0
    region_exp: dict[str, float] = {}
    sleeve_exp: dict[str, float] = {}
    currency_exp: dict[str, float] = {}
    eligible_keys: set[str] = set()
    for index in range(1, len(dates)):
        target = weights_by_period[index] if index < len(weights_by_period) else previous_weights
        eligible_keys.update(key for key, weight in target.items() if key in metadata and weight > 0)
        nav_before = nav
        turnover = 0.5 * sum(abs(target.get(key, 0.0) - previous_weights.get(key, 0.0)) for key in set(target) | set(previous_weights))
        cost_rate = turnover * cost_bps / 10_000.0
        gross_return = 0.0
        asset_contrib: dict[str, float] = {}
        for key, weight in previous_weights.items():
            if key not in returns:
                continue
            contribution = weight * returns[key][index]
            gross_return += contribution
            pnl = contribution * nav_before
            asset_contrib[key] = pnl
            contribution_by_asset[key] = contribution_by_asset.get(key, 0.0) + pnl
            contribution_by_year[dates[index][:4]] = contribution_by_year.get(dates[index][:4], 0.0) + pnl
            if key in active:
                active[key]["pnl"] += pnl
        cost_pnl = cost_rate * nav_before
        net_return = gross_return - cost_rate
        nav = nav_before * (1.0 + net_return)
        gross_pnl += gross_return * nav_before
        cost_amount += cost_pnl
        turnover_amount += turnover * nav_before
        navs.append(nav)
        net_returns.append(net_return)
        turnovers.append(turnover)
        costs.append(cost_rate)
        cash_weight = max(0.0, 1.0 - sum(weight for key, weight in previous_weights.items() if key in returns))
        cash_weights.append(cash_weight)
        _accumulate_exposure(region_exp, sleeve_exp, currency_exp, previous_weights, metadata)
        for key in list(active):
            if target.get(key, 0.0) <= 0:
                episode = active.pop(key)
                episodes.append(
                    {
                        "con_id": int(key),
                        "symbol": metadata[key]["symbol"],
                        "start_date": episode["start_date"],
                        "end_date": dates[index],
                        "pnl": episode["pnl"],
                    }
                )
        for key, weight in target.items():
            if key in metadata and weight > 0 and previous_weights.get(key, 0.0) <= 0:
                active[key] = {"start_date": dates[index], "pnl": 0.0}
        daily.append(
            {
                "date": dates[index],
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "cost": cost_rate,
                "cash_weight": cash_weight,
                "nav": nav,
                "asset_contribution": asset_contrib,
            }
        )
        previous_weights = target
    for key, episode in list(active.items()):
        episodes.append(
            {
                "con_id": int(key),
                "symbol": metadata[key]["symbol"],
                "start_date": episode["start_date"],
                "end_date": dates[-1] if dates else None,
                "pnl": episode["pnl"],
            }
        )
    metrics: dict[str, Any] = {
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        "instrument_count": len(returns),
        "rebalance_count": sum(1 for value in turnovers if value > 0),
        "closed_position_episodes": len(episodes),
        "gross_return": navs[-1] / navs[0] - 1.0 + sum(costs) if len(navs) > 1 else 0.0,
        "net_return": navs[-1] / navs[0] - 1.0 if len(navs) > 1 else 0.0,
        "CAGR": _cagr(dates[0] if dates else None, dates[-1] if dates else None, navs[-1] / navs[0] - 1.0 if len(navs) > 1 else 0.0),
        "annualized_volatility": _annualized_vol(net_returns),
        "Sharpe": _sharpe(net_returns),
        "Sortino": _sortino(net_returns),
        "maximum_drawdown": _max_drawdown(navs),
        "Calmar": None,
        "trade_profit_factor": _profit_factor([episode["pnl"] for episode in episodes]),
        "period_profit_factor": _profit_factor(net_returns),
        "expectancy_per_episode": _mean([episode["pnl"] for episode in episodes]) if episodes else None,
        "win_rate": sum(1 for episode in episodes if episode["pnl"] > 0) / len(episodes) if episodes else None,
        "average_win": _mean([episode["pnl"] for episode in episodes if episode["pnl"] > 0]) if any(episode["pnl"] > 0 for episode in episodes) else None,
        "average_loss": _mean([episode["pnl"] for episode in episodes if episode["pnl"] < 0]) if any(episode["pnl"] < 0 for episode in episodes) else None,
        "turnover": sum(turnovers),
        "transaction_costs": sum(costs),
        "average_cash": _mean(cash_weights),
        "maximum_cash": max(cash_weights) if cash_weights else 0.0,
        "region_exposure": {key: value / max(1, len(daily)) for key, value in sorted(region_exp.items())},
        "sleeve_exposure": {key: value / max(1, len(daily)) for key, value in sorted(sleeve_exp.items())},
        "currency_exposure": {key: value / max(1, len(daily)) for key, value in sorted(currency_exp.items())},
    }
    metrics["Calmar"] = None if not metrics["CAGR"] or metrics["maximum_drawdown"] == 0 else metrics["CAGR"] / abs(metrics["maximum_drawdown"])
    return {
        "schema": "phase6_1_detailed_portfolio_v1",
        "name": name,
        "parameters": parameters,
        "metrics": metrics,
        "daily": daily,
        "episodes": episodes,
        "episode_summary": {
            "positive_episodes": sum(1 for episode in episodes if episode["pnl"] > 0),
            "negative_episodes": sum(1 for episode in episodes if episode["pnl"] < 0),
            "zero_pnl_episodes": sum(1 for episode in episodes if episode["pnl"] == 0),
        },
        "eligible_instruments": sorted({metadata[key]["symbol"] for key in eligible_keys}),
        "gross_pnl": gross_pnl,
        "net_pnl": nav - 100.0,
        "cost_amount": cost_amount,
        "turnover_amount": turnover_amount,
        "contribution_by_asset": contribution_by_asset,
        "contribution_by_year": contribution_by_year,
        "financial_calls": _zero_financial_calls(),
    }


def _config_candidates() -> list[dict[str, Any]]:
    return [
        {
            "momentum_lookback": momentum,
            "trend_lookback": trend,
            "rebalance": rebalance,
            "target_vol": target_vol,
            "cost_bps": cost_bps,
        }
        for momentum, trend, rebalance, target_vol, cost_bps in itertools.product(
            (63, 126, 252),
            (100, 200),
            ("weekly", "monthly"),
            (0.08, 0.10, 0.12),
            (5.0, 10.0, 20.0),
        )
    ]


def _targets_for_config(prepared: dict[str, Any], config: dict[str, Any]) -> list[dict[str, float]]:
    return _momentum_targets(
        prepared,
        momentum_lookback=int(config["momentum_lookback"]),
        trend_lookback=int(config["trend_lookback"]),
        top_n=4,
        rebalance=str(config["rebalance"]),
        target_vol=float(config["target_vol"]),
    )


def _load_phase6_grid(project_root: Path) -> dict[str, Any]:
    path = Phase6Layout.from_project_root(project_root).strategy_grid_json
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    dataset = load_phase6_dataset(project_root)
    from stocks.research.phase6 import run_strategy_grid

    return run_strategy_grid(dataset)


def _canonical_config(grid: dict[str, Any]) -> dict[str, Any]:
    best = grid.get("best_by_calmar", {})
    params = best.get("parameters") or {}
    return {
        "momentum_lookback": int(params.get("momentum_lookback", 252)),
        "trend_lookback": int(params.get("trend_lookback", 100)),
        "rebalance": str(params.get("rebalance", "monthly")),
        "target_vol": float(params.get("target_vol", 0.12)),
        "cost_bps": float(params.get("cost_bps", 10.0)),
    }


def _pf_diagnostics(values: list[float], *, sample_name: str) -> dict[str, Any]:
    positives = [value for value in values if value > 0]
    negatives = [value for value in values if value < 0]
    zeros = [value for value in values if value == 0]
    pf = _profit_factor(values)
    reason: str | None
    if not values:
        reason = "NO_TRADES" if sample_name == "episode" else "METRIC_ERROR"
    elif len(values) < 10:
        reason = "INSUFFICIENT_EPISODES" if sample_name == "episode" else None
    elif not positives and negatives:
        reason = "ALL_TRADES_LOSS" if sample_name == "episode" else "NO_POSITIVE_EPISODES"
    elif positives and not negatives:
        reason = "ZERO_DENOMINATOR"
    else:
        reason = None
    return {
        "sample": sample_name,
        "count": len(values),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "zero_count": len(zeros),
        "gross_positive": sum(positives),
        "gross_negative_abs": abs(sum(negatives)),
        "profit_factor": pf,
        "diagnostic_reason": reason,
        "sample_status": "INSUFFICIENT_SAMPLE" if sample_name == "episode" and len(values) < 10 else "EVALUABLE",
    }


def _pf_zero_reason(detailed: dict[str, Any]) -> str | None:
    period = _pf_diagnostics([row["net_return"] for row in detailed["daily"]], sample_name="period")
    episode = _pf_diagnostics([item["pnl"] for item in detailed["episodes"]], sample_name="episode")
    if not detailed["episodes"] and (period["profit_factor"] == 0 or episode["profit_factor"] == 0):
        return "NO_TRADES"
    if period["profit_factor"] == 0:
        return str(period["diagnostic_reason"] or "METRIC_ERROR")
    if episode["profit_factor"] == 0:
        return str(episode["diagnostic_reason"] or "METRIC_ERROR")
    return None


def _decision_status(
    folds: dict[str, Any] | None,
    benchmark_comparison: dict[str, Any] | None,
    concentration: dict[str, Any] | None,
    costs: dict[str, Any] | None,
    plateau: dict[str, Any] | None,
    multiple: dict[str, Any] | None,
    monte_carlo: dict[str, Any] | None,
    sample: dict[str, Any] | None,
) -> str:
    if (
        folds is None
        or benchmark_comparison is None
        or concentration is None
        or costs is None
        or plateau is None
        or multiple is None
        or monte_carlo is None
        or sample is None
    ):
        return "METRIC_IMPLEMENTATION_BLOCKED"
    if sample.get("status") == "INSUFFICIENT_SAMPLE":
        return "INSUFFICIENT_SAMPLE"
    fold_count = folds.get("fold_count", 0)
    positive_folds = folds.get("positive_folds", 0)
    median_episode_pf = _median([fold["episode_profit_factor"] for fold in folds.get("folds", []) if fold["episode_profit_factor"] is not None])
    stress_rows = costs.get("results", [])
    stress_20: dict[str, Any] = next(
        (
            row
            for row in stress_rows
            if isinstance(row, dict) and row.get("cost_bps") == 20.0
        ),
        {},
    )
    raw_dominance = concentration.get("dominance_flags", {})
    dominance = raw_dominance if isinstance(raw_dominance, dict) else {}
    benchmark_pass = benchmark_comparison.get("strategy_beats_benchmark_count", 0) > benchmark_comparison.get("comparison_count", 1) / 2
    finalist = (
        positive_folds >= 4
        and (median_episode_pf or 0) > 1.10
        and (folds.get("worst_period_pf") or 0) > 0.80
        and (stress_20.get("period_profit_factor") or 0) > 1.0
        and benchmark_pass
        and not any(dominance.values())
        and plateau.get("has_plateau") is True
        and (multiple.get("probability_of_backtest_overfitting") or 1.0) < 0.5
        and (multiple.get("deflated_sharpe_probability") or 0.0) > 0.5
        and (monte_carlo.get("probability_PF_gt_1") or 0.0) > 0.5
    )
    if finalist:
        return "FINANCIAL_FINALIST_GO"
    if positive_folds >= max(1, math.ceil(fold_count * 0.5)) and (folds.get("worst_period_pf") == 0 or (folds.get("worst_period_pf") or 0) < 0.80):
        return "PROMISING_RESEARCH_CANDIDATE"
    return "NO_FINANCIAL_FINALIST"


def _sample_status(episodes: int, active_months: int, rebalances: int, observations: int) -> str:
    if episodes < 10 or active_months < 6 or rebalances < 2 or observations < 63:
        return "INSUFFICIENT_SAMPLE"
    if episodes < 30:
        return "LOW_CONFIDENCE"
    return "EVALUABLE"


def _world_index_key(prepared: dict[str, Any]) -> str:
    for key, meta in prepared["metadata"].items():
        symbol = str(meta["symbol"]).upper()
        instrument_id = str(meta["instrument_id"]).upper()
        if symbol in {"ACWI", "URTH", "IWDA", "SPY"} or "WORLD" in instrument_id:
            return key
    return sorted(prepared["returns"])[0]


def _excess_cagr(excess: list[float], dates: list[str]) -> float | None:
    if not excess or len(dates) < 2:
        return None
    return _cagr(dates[1], dates[-1], _compound_return(excess))


def _information_ratio(excess: list[float]) -> float | None:
    std = _std(excess)
    return None if std == 0 else _mean(excess) / std * math.sqrt(252)


def _sharpe(values: list[float]) -> float | None:
    std = _std(values)
    return None if std == 0 else _mean(values) / std * math.sqrt(252)


def _sortino(values: list[float]) -> float | None:
    downside = [min(0.0, value) for value in values]
    std = _std(downside)
    return None if std == 0 else _mean(values) / std * math.sqrt(252)


def _compound_return(values: list[float]) -> float:
    nav = 1.0
    for value in values:
        nav *= 1.0 + value
    return nav - 1.0


def _nav_path(values: list[float]) -> list[float]:
    nav = 100.0
    path = [nav]
    for value in values:
        nav *= 1.0 + value
        path.append(nav)
    return path


def _positive_share(mapping: dict[str, float]) -> dict[str, float]:
    positives = {key: max(0.0, value) for key, value in mapping.items()}
    total = sum(positives.values())
    if total == 0:
        return {key: 0.0 for key in sorted(mapping)}
    return {key: value / total for key, value in sorted(positives.items())}


def _group_contribution(mapping: dict[str, float], metadata: dict[str, dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for key, value in mapping.items():
        group = str(metadata.get(key, {}).get(field, "unknown"))
        grouped[group] = grouped.get(group, 0.0) + value
    return dict(sorted(grouped.items()))


def _symbolized(mapping: dict[str, float], metadata: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {str(metadata.get(key, {}).get("symbol", key)): value for key, value in sorted(mapping.items(), key=lambda item: str(metadata.get(item[0], {}).get("symbol", item[0])))}


def _remove_keys(prepared: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    returns = {key: value for key, value in prepared["returns"].items() if key not in keys}
    prices = {key: value for key, value in prepared["prices"].items() if key not in keys}
    metadata = {key: value for key, value in prepared["metadata"].items() if key not in keys}
    return {**prepared, "returns": returns, "prices": prices, "metadata": metadata, "cash_key": _cash_key(metadata)}


def _nullable_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _regime_labels(prepared: dict[str, Any], dataset: dict[str, Any]) -> dict[str, set[str]]:
    dates = prepared["dates"]
    eq = [sum(values[index] for values in prepared["returns"].values()) / len(prepared["returns"]) for index in range(len(dates))]
    vol_values = [_std(eq[max(0, index - 63) : index]) for index in range(len(eq))]
    vol_median = _median(vol_values) or 0.0
    bond_keys = [key for key, meta in prepared["metadata"].items() if meta["sleeve"] == "bond"]
    commodity_keys = [key for key, meta in prepared["metadata"].items() if meta["sleeve"] == "commodity"]
    usd_fx = _usd_fx_returns_by_date(dataset)
    labels: dict[str, set[str]] = {}
    for index, day in enumerate(dates):
        lookback = eq[max(0, index - 63) : index + 1]
        trend = _compound_return(lookback)
        bucket = labels.setdefault(day, set())
        bucket.add("bull_market" if trend >= 0 else "bear_market")
        bucket.add("high_volatility" if vol_values[index] >= vol_median else "low_volatility")
        bond_return = _basket_return(prepared, bond_keys, index)
        bucket.add("rising_rates" if bond_return < 0 else "falling_rates")
        commodity_return = _basket_return(prepared, commodity_keys, index)
        bucket.add("inflationary" if commodity_return > 0 else "disinflationary")
        bucket.add("USD_strengthening" if usd_fx.get(day, 0.0) > 0 else "USD_weakening")
    return labels


def _basket_return(prepared: dict[str, Any], keys: list[str], index: int) -> float:
    if not keys:
        return 0.0
    return sum(prepared["returns"][key][index] for key in keys) / len(keys)


def _usd_fx_returns_by_date(dataset: dict[str, Any]) -> dict[str, float]:
    for records in dataset["series"].values():
        if records and str(records[0].get("instrument_currency")) == "USD":
            return {str(record["session_date"]): float(record.get("fx_return") or 0.0) for record in records}
    return {}


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _block_bootstrap(values: list[float], *, block_size: int, rng: random.Random) -> list[float]:
    if not values:
        return []
    sampled: list[float] = []
    while len(sampled) < len(values):
        start = rng.randrange(0, max(1, len(values) - block_size + 1))
        sampled.extend(values[start : start + block_size])
    return sampled[: len(values)]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _start_date_perturbation(returns: list[float]) -> dict[str, Any]:
    rows = []
    for offset in (0, 21, 42, 63):
        path = returns[offset:] if offset < len(returns) else []
        rows.append({"offset_days": offset, "terminal_NAV": _nav_path(path)[-1] if path else None, "maximum_drawdown": _max_drawdown(_nav_path(path)) if path else None})
    return {"offsets_tested": len(rows), "results": rows}


def _effective_sample_size(values: list[float]) -> float:
    if len(values) < 3:
        return float(len(values))
    avg = _mean(values)
    numerator = sum((left - avg) * (right - avg) for left, right in zip(values, values[1:], strict=False))
    denominator = sum((value - avg) ** 2 for value in values)
    autocorr = 0.0 if denominator == 0 else numerator / denominator
    return len(values) / max(1e-9, 1.0 + 2.0 * max(0.0, autocorr))


def _accumulate_exposure(
    region_acc: dict[str, float],
    sleeve_acc: dict[str, float],
    currency_acc: dict[str, float],
    weights: dict[str, float],
    metadata: dict[str, dict[str, Any]],
) -> None:
    for key, weight in weights.items():
        if key not in metadata:
            continue
        meta = metadata[key]
        region_acc[str(meta["region"])] = region_acc.get(str(meta["region"]), 0.0) + weight
        sleeve_acc[str(meta["sleeve"])] = sleeve_acc.get(str(meta["sleeve"]), 0.0) + weight
        currency_acc[str(meta["currency"])] = currency_acc.get(str(meta["currency"]), 0.0) + weight


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
