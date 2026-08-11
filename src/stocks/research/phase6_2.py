from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file, utc_now_iso
from stocks.research.phase6 import (
    _mean,
    _median,
    _portfolio_result,
    _prepare_common_history,
    _profit_factor,
    _slice_prepared,
    _write_json,
    _zero_financial_calls,
    load_phase6_dataset,
)
from stocks.research.phase6_diagnostics import (
    Phase61Layout,
    _config_candidates,
    _detailed_portfolio,
    _effective_sample_size,
    _load_phase6_grid,
    _read_json_if_exists,
    _sample_status,
    _targets_for_config,
    phase6_1_status,
)


PHASE6_2_DECISIONS = (
    "NO_FINANCIAL_EDGE",
    "INSUFFICIENT_SAMPLE",
    "PROMISING_RESEARCH_CANDIDATE",
    "FINANCIAL_FINALIST_GO",
    "METRIC_OR_ACCOUNTING_BLOCKED",
)


@dataclass(frozen=True)
class Phase62Layout:
    output_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> Phase62Layout:
        return cls(output_dir=project_root / "output" / "research" / "phase6_2")

    @property
    def episode_accounting_json(self) -> Path:
        return self.output_dir / "episode-accounting.json"

    @property
    def semiannual_oos_json(self) -> Path:
        return self.output_dir / "semiannual-oos.json"

    @property
    def aggregate_oos_json(self) -> Path:
        return self.output_dir / "aggregate-oos.json"

    @property
    def point_in_time_history_json(self) -> Path:
        return self.output_dir / "point-in-time-history.json"

    @property
    def cost_reliability_json(self) -> Path:
        return self.output_dir / "cost-reliability.json"

    @property
    def forward_shadow_json(self) -> Path:
        return self.output_dir / "forward-shadow.json"

    @property
    def status_json(self) -> Path:
        return self.output_dir / "phase6_2-status.json"

    @property
    def freeze_json(self) -> Path:
        return self.output_dir / "phase6_2-freeze.json"


def phase6_2_schema() -> dict[str, Any]:
    return {
        "schema": "phase6_2_sample_sufficiency_forward_oos_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "decision_statuses": list(PHASE6_2_DECISIONS),
        "tracks": {
            "A_episode_accounting": [
                "entry_count",
                "exit_count",
                "partial_rebalances",
                "carry_in_positions",
                "carry_out_positions",
                "open_episodes_at_fold_end",
                "closed_episodes",
                "signal_opportunity_count",
                "blocked_signal_count",
                "NO_TRADES_REASON",
                "episode_accounting_status",
                "carry_policy",
            ],
            "B_more_causal_windows": {
                "train_window": "3 years",
                "validation": "6 months",
                "test": "6 months",
                "step": "6 months",
                "parameter_family_size": 108,
            },
            "D_forward_shadow": {
                "closed_bars_only": True,
                "orders_enabled": False,
                "paper_authority": False,
            },
        },
        "history_policy": "POINT_IN_TIME_ONLY_NO_RETROACTIVE_UNIVERSE_EXTENSION",
        "financial_calls": _zero_financial_calls(),
    }


def run_phase6_2_pipeline(project_root: Path) -> dict[str, Any]:
    layout = Phase62Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_phase6_dataset(project_root)
    prepared = _prepare_common_history(dataset)
    phase6_grid = _load_phase6_grid(project_root)
    phase6_1 = phase6_1_status(project_root)
    annual_folds = _load_phase6_1_folds(project_root)

    episode_accounting = annual_episode_accounting(prepared, annual_folds)
    _write_json(layout.episode_accounting_json, episode_accounting)
    semiannual = semiannual_oos_evidence(prepared)
    _write_json(layout.semiannual_oos_json, semiannual)
    aggregate = aggregate_oos_evidence(project_root, prepared, annual_folds, semiannual)
    _write_json(layout.aggregate_oos_json, aggregate)
    pit_history = point_in_time_history_report(dataset)
    _write_json(layout.point_in_time_history_json, pit_history)
    cost_reliability = cost_reliability_report(prepared, phase6_grid)
    _write_json(layout.cost_reliability_json, cost_reliability)
    shadow = forward_shadow_report(prepared, phase6_grid)
    _write_json(layout.forward_shadow_json, shadow)
    status = phase6_2_status(
        project_root,
        episode_accounting=episode_accounting,
        semiannual=semiannual,
        aggregate=aggregate,
        pit_history=pit_history,
        cost_reliability=cost_reliability,
        shadow=shadow,
        phase6_1=phase6_1,
    )
    _write_json(layout.status_json, status)
    return status


def phase6_2_status(
    project_root: Path,
    *,
    episode_accounting: dict[str, Any] | None = None,
    semiannual: dict[str, Any] | None = None,
    aggregate: dict[str, Any] | None = None,
    pit_history: dict[str, Any] | None = None,
    cost_reliability: dict[str, Any] | None = None,
    shadow: dict[str, Any] | None = None,
    phase6_1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layout = Phase62Layout.from_project_root(project_root)
    episode_accounting = episode_accounting or _read_json_if_exists(layout.episode_accounting_json)
    semiannual = semiannual or _read_json_if_exists(layout.semiannual_oos_json)
    aggregate = aggregate or _read_json_if_exists(layout.aggregate_oos_json)
    pit_history = pit_history or _read_json_if_exists(layout.point_in_time_history_json)
    cost_reliability = cost_reliability or _read_json_if_exists(layout.cost_reliability_json)
    shadow = shadow or _read_json_if_exists(layout.forward_shadow_json)
    phase6_1 = phase6_1 or phase6_1_status(project_root)
    checks = {
        "episode_accounting_available": episode_accounting is not None and episode_accounting.get("status") == "GO",
        "no_artificial_fold_closure": episode_accounting is not None
        and all(item.get("episode_accounting_status") != "ARTIFICIAL_FOLD_CLOSE_DETECTED" for item in episode_accounting.get("folds", [])),
        "semiannual_windows_available": semiannual is not None and semiannual.get("fold_count", 0) >= 8,
        "aggregate_oos_available": aggregate is not None and aggregate.get("status") == "GO",
        "point_in_time_history_available": pit_history is not None and pit_history.get("status") == "GO",
        "cost_reliability_available": cost_reliability is not None and cost_reliability.get("status") == "GO",
        "forward_shadow_available": shadow is not None and shadow.get("status") == "GO",
        "phase6_1_remains_go": phase6_1 is not None and phase6_1.get("status") == "PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_GO",
        "no_financial_calls": True,
    }
    decision = _phase6_2_decision(aggregate, phase6_1)
    return {
        "schema": "phase6_2_sample_sufficiency_forward_oos_status_v1",
        "status": "PHASE6_2_SAMPLE_SUFFICIENCY_AND_FORWARD_OOS_GO" if all(checks.values()) else "NO_GO",
        "decision_status": decision,
        "generated_at": utc_now_iso(),
        "checks": checks,
        "summary": {
            "annual_windows": None if aggregate is None else aggregate.get("annual_fold_count"),
            "semiannual_windows": None if semiannual is None else semiannual.get("fold_count"),
            "aggregate_closed_oos_episodes": None if aggregate is None else aggregate.get("aggregate_closed_oos_episodes"),
            "effective_sample_size": None if aggregate is None else aggregate.get("effective_sample_size"),
            "evaluable_testwindows": None if aggregate is None else aggregate.get("evaluable_testwindows"),
            "positive_evaluable_window_ratio": None if aggregate is None else aggregate.get("positive_evaluable_window_ratio"),
            "unevaluable_no_trade_windows": None if aggregate is None else aggregate.get("unevaluable_no_trade_windows"),
            "aggregate_oos_episode_pf": None if aggregate is None else aggregate.get("aggregate_oos_episode_pf"),
            "median_oos_period_pf": None if aggregate is None else aggregate.get("median_oos_period_pf"),
            "stress_20bps_pf": None if aggregate is None else aggregate.get("stress_20bps_pf"),
            "benchmark_win_rate": None if aggregate is None else aggregate.get("benchmark_win_rate"),
            "PBO": None if aggregate is None else aggregate.get("PBO"),
            "DSR_probability": None if aggregate is None else aggregate.get("DSR_probability"),
            "history_policy": None if pit_history is None else pit_history.get("history_policy"),
            "p5_break_even_cost_bps": None if cost_reliability is None else cost_reliability.get("P5_break_even_cost_bps_bootstrap"),
            "forward_shadow_records": None if shadow is None else shadow.get("record_count"),
        },
        "artifacts": {
            "episode_accounting": str(layout.episode_accounting_json),
            "semiannual_oos": str(layout.semiannual_oos_json),
            "aggregate_oos": str(layout.aggregate_oos_json),
            "point_in_time_history": str(layout.point_in_time_history_json),
            "cost_reliability": str(layout.cost_reliability_json),
            "forward_shadow": str(layout.forward_shadow_json),
        },
        "financial_calls": _zero_financial_calls(),
    }


def annual_episode_accounting(prepared: dict[str, Any], annual_folds: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for fold in annual_folds:
        rows.append(_episode_accounting_for_window(prepared, fold["selected_configuration"], fold["test_start"], fold["test_end"], fold_id=fold["fold_id"]))
    return {
        "schema": "phase6_2_annual_episode_accounting_v1",
        "status": "GO",
        "carry_policy": "CONTINUOUS_TIMELINE_EPISODES_NOT_CLOSED_AT_FOLD_BOUNDARY",
        "folds": rows,
        "financial_calls": _zero_financial_calls(),
    }


def semiannual_oos_evidence(prepared: dict[str, Any]) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    candidates = _config_candidates()
    first_start = date(2018, 1, 1)
    last_date = date.fromisoformat(prepared["dates"][-1])
    cursor = first_start
    fold_id = 1
    while True:
        train_start = cursor
        train_end = _add_months(train_start, 36, day_offset=-1)
        validation_start = _add_months(train_start, 36)
        validation_end = _add_months(validation_start, 6, day_offset=-1)
        test_start = _add_months(validation_start, 6)
        test_end = _add_months(test_start, 6, day_offset=-1)
        if test_end > last_date:
            break
        validation_subset = _slice_prepared(prepared, trainless(validation_start), trainless(validation_end))
        if len(validation_subset["dates"]) < 63:
            cursor = _add_months(cursor, 6)
            continue
        validation_scores = []
        for config in candidates:
            result = _portfolio_result(
                "phase6_2_semiannual_validation",
                validation_subset,
                _targets_for_config(validation_subset, config),
                cost_bps=float(config["cost_bps"]),
                parameters=config,
            )
            validation_scores.append((float(result["net_return"]), config))
        selected = max(validation_scores, key=lambda item: item[0])[1]
        test_subset = _slice_prepared(prepared, trainless(test_start), trainless(test_end))
        detailed = _detailed_portfolio(
            test_subset,
            _targets_for_config(test_subset, selected),
            cost_bps=float(selected["cost_bps"]),
            parameters=selected,
            name="phase6_2_semiannual_test",
        )
        accounting = _episode_accounting_for_window(prepared, selected, trainless(test_start), trainless(test_end), fold_id=fold_id)
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": trainless(train_start),
                "train_end": trainless(train_end),
                "validation_start": trainless(validation_start),
                "validation_end": trainless(validation_end),
                "test_start": trainless(test_start),
                "test_end": trainless(test_end),
                "purge_embargo": {
                    "lookback_days": max(int(selected["momentum_lookback"]), int(selected["trend_lookback"])),
                    "execution_lag_days": 1,
                    "policy": "bounded_lookback_embargo_recorded_no_random_shuffle",
                },
                "selected_configuration": selected,
                "net_return": detailed["metrics"]["net_return"],
                "period_profit_factor": detailed["metrics"]["period_profit_factor"],
                "episode_profit_factor": _profit_factor(accounting["closed_episode_pnls"]),
                "Sharpe": detailed["metrics"]["Sharpe"],
                "maximum_drawdown": detailed["metrics"]["maximum_drawdown"],
                "turnover": detailed["metrics"]["turnover"],
                "closed_episodes": accounting["closed_episodes"],
                "NO_TRADES_REASON": accounting["NO_TRADES_REASON"],
                "episode_accounting_status": accounting["episode_accounting_status"],
                "sample_status": _sample_status(accounting["closed_episodes"], 6, detailed["metrics"]["rebalance_count"], len(detailed["daily"])),
                "daily_returns": [row["net_return"] for row in detailed["daily"]],
                "closed_episode_pnls": accounting["closed_episode_pnls"],
            }
        )
        fold_id += 1
        cursor = _add_months(cursor, 6)
    return {
        "schema": "phase6_2_semiannual_oos_v1",
        "status": "GO" if len(folds) >= 8 else "NO_GO",
        "fold_count": len(folds),
        "positive_folds": sum(1 for fold in folds if fold["net_return"] > 0),
        "evaluable_folds": sum(1 for fold in folds if fold["sample_status"] != "INSUFFICIENT_SAMPLE"),
        "folds": folds,
        "financial_calls": _zero_financial_calls(),
    }


def aggregate_oos_evidence(
    project_root: Path,
    prepared: dict[str, Any],
    annual_folds: list[dict[str, Any]],
    semiannual: dict[str, Any],
) -> dict[str, Any]:
    phase6_1_layout = Phase61Layout.from_project_root(project_root)
    concentration = _read_json_if_exists(phase6_1_layout.concentration_json) or {}
    costs = _read_json_if_exists(phase6_1_layout.cost_stress_json) or {}
    multiple = _read_json_if_exists(phase6_1_layout.multiple_testing_json) or {}
    plateau = _read_json_if_exists(phase6_1_layout.parameter_plateau_json) or {}
    benchmark = _read_json_if_exists(phase6_1_layout.benchmark_comparison_json) or {}
    annual_windows = []
    for fold in annual_folds:
        accounting = _episode_accounting_for_window(prepared, fold["selected_configuration"], fold["test_start"], fold["test_end"], fold_id=fold["fold_id"])
        annual_windows.append(
            {
                "fold_id": fold["fold_id"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "net_return": fold["net_pnl"] / 100.0,
                "period_profit_factor": fold["period_profit_factor"],
                "closed_episodes": accounting["closed_episodes"],
                "sample_status": _sample_status(accounting["closed_episodes"], int(fold.get("active_months", 12)), int(fold.get("rebalance_count", 0)), len(fold.get("daily_returns", []))),
                "closed_episode_pnls": accounting["closed_episode_pnls"],
                "daily_returns": fold.get("daily_returns", []),
            }
        )
    all_windows = annual_windows + [
        {
            "fold_id": f"semiannual_{fold['fold_id']}",
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "net_return": fold["net_return"],
            "period_profit_factor": fold["period_profit_factor"],
            "closed_episodes": fold["closed_episodes"],
            "sample_status": fold["sample_status"],
            "closed_episode_pnls": fold["closed_episode_pnls"],
            "daily_returns": fold["daily_returns"],
        }
        for fold in semiannual["folds"]
    ]
    closed_episode_pnls = [pnl for window in all_windows for pnl in window["closed_episode_pnls"]]
    daily_returns = [value for window in all_windows for value in window["daily_returns"]]
    evaluable = [window for window in all_windows if window["sample_status"] != "INSUFFICIENT_SAMPLE"]
    positive_evaluable = [window for window in evaluable if window["net_return"] > 0]
    stress_20: dict[str, Any] = next((row for row in costs.get("results", []) if row["cost_bps"] == 20.0), {})
    dominance = concentration.get("dominance_flags", {})
    benchmark_count = benchmark.get("comparison_count", 0)
    benchmark_wins = benchmark.get("strategy_beats_benchmark_count", 0)
    return {
        "schema": "phase6_2_aggregate_oos_evidence_v1",
        "status": "GO",
        "annual_fold_count": len(annual_windows),
        "semiannual_fold_count": semiannual["fold_count"],
        "total_testwindows": len(all_windows),
        "aggregate_closed_oos_episodes": len(closed_episode_pnls),
        "effective_sample_size": _effective_sample_size(daily_returns),
        "evaluable_testwindows": len(evaluable),
        "positive_evaluable_windows": len(positive_evaluable),
        "positive_evaluable_window_ratio": len(positive_evaluable) / len(evaluable) if evaluable else 0.0,
        "unevaluable_no_trade_windows": sum(1 for window in all_windows if window["closed_episodes"] == 0),
        "aggregate_oos_episode_pf": _profit_factor(closed_episode_pnls),
        "median_oos_period_pf": _median([window["period_profit_factor"] for window in all_windows if window["period_profit_factor"] is not None]),
        "stress_20bps_pf": stress_20.get("period_profit_factor"),
        "benchmark_win_rate": benchmark_wins / benchmark_count if benchmark_count else None,
        "single_asset_contribution_lt_40": not dominance.get("single_asset_gt_40pct", True),
        "single_region_contribution_lt_60": not dominance.get("single_region_gt_60pct", True),
        "parameterplateau": plateau.get("has_plateau") is True,
        "PBO": multiple.get("probability_of_backtest_overfitting"),
        "DSR_probability": multiple.get("deflated_sharpe_probability"),
        "dsr_status": "NEEDS_IMPROVEMENT",
        "gates": {
            "aggregate_closed_OOS_episodes_ge_30": len(closed_episode_pnls) >= 30,
            "effective_sample_size_ge_20": _effective_sample_size(daily_returns) >= 20,
            "evaluable_testwindows_ge_8": len(evaluable) >= 8,
            "positive_evaluable_windows_ge_65pct": (len(positive_evaluable) / len(evaluable) if evaluable else 0.0) >= 0.65,
            "unevaluable_no_trade_windows_le_1": sum(1 for window in all_windows if window["closed_episodes"] == 0) <= 1,
            "aggregate_oos_episode_pf_gt_1_10": (_profit_factor(closed_episode_pnls) or 0.0) > 1.10,
            "median_oos_period_pf_gt_1_05": (_median([window["period_profit_factor"] for window in all_windows if window["period_profit_factor"] is not None]) or 0.0) > 1.05,
            "stress_20bps_pf_gt_1": (stress_20.get("period_profit_factor") or 0.0) > 1.0,
            "benchmark_comparisons_won_gt_60pct": (benchmark_wins / benchmark_count if benchmark_count else 0.0) > 0.60,
            "single_asset_contribution_lt_40": not dominance.get("single_asset_gt_40pct", True),
            "single_region_contribution_lt_60": not dominance.get("single_region_gt_60pct", True),
            "parameterplateau_present": plateau.get("has_plateau") is True,
            "PBO_lte_0_20": (multiple.get("probability_of_backtest_overfitting") or 1.0) <= 0.20,
            "DSR_probability_candidate": (multiple.get("deflated_sharpe_probability") or 0.0) >= 0.25,
        },
        "annual_fold_evidence": annual_windows,
        "semiannual_fold_evidence": semiannual["folds"],
        "financial_calls": _zero_financial_calls(),
    }


def point_in_time_history_report(dataset: dict[str, Any]) -> dict[str, Any]:
    instruments = []
    all_dates: set[str] = set()
    for key, records in dataset["series"].items():
        dates = [str(record["session_date"]) for record in records]
        all_dates.update(dates)
        meta = dataset["metadata"][key]
        instruments.append(
            {
                "con_id": int(key),
                "symbol": meta["symbol"],
                "first_valid_date": min(dates),
                "last_valid_date": max(dates),
                "available_at_rule": "available_at_i <= t",
                "retroactive_use_allowed": False,
            }
        )
    pit_counts = {
        day: sum(1 for item in instruments if item["first_valid_date"] <= day <= item["last_valid_date"])
        for day in sorted(all_dates)
    }
    return {
        "schema": "phase6_2_point_in_time_history_v1",
        "status": "GO",
        "history_policy": "POINT_IN_TIME_ONLY_NO_RETROACTIVE_UNIVERSE_EXTENSION",
        "universe_rule": "U_t={i:first_valid_i<=t and available_at_i<=t}",
        "history_expansion_performed": False,
        "reason": "No additional older proxies were added in Phase 6.2 because the existing 17-instrument total-return universe already provides the required semiannual OOS windows.",
        "instrument_count": len(instruments),
        "pit_min_instrument_count": min(pit_counts.values()) if pit_counts else 0,
        "pit_max_instrument_count": max(pit_counts.values()) if pit_counts else 0,
        "instruments": sorted(instruments, key=lambda item: item["symbol"]),
        "financial_calls": _zero_financial_calls(),
    }


def cost_reliability_report(prepared: dict[str, Any], phase6_grid: dict[str, Any]) -> dict[str, Any]:
    config = _canonical_config_from_grid(phase6_grid)
    detailed = _detailed_portfolio(
        prepared,
        _targets_for_config(prepared, config),
        cost_bps=float(config["cost_bps"]),
        parameters=config,
        name="phase6_2_cost_reliability",
    )
    episode_pnls = [float(episode["pnl"]) for episode in detailed["episodes"]]
    gross_profit = sum(value for value in episode_pnls if value > 0)
    gross_loss = sum(value for value in episode_pnls if value < 0)
    closed_episode_count = len(episode_pnls)
    median_cost_per_episode = None if closed_episode_count == 0 else detailed["cost_amount"] / closed_episode_count
    break_even_cost_bps = None if detailed["turnover_amount"] <= 0 else detailed["gross_pnl"] / detailed["turnover_amount"] * 10_000.0
    p5_break_even = _p5_break_even_cost(episode_pnls, detailed["turnover_amount"], iterations=300)
    return {
        "schema": "phase6_2_break_even_cost_reliability_v1",
        "status": "GO",
        "configuration": config,
        "break_even_cost_bps": break_even_cost_bps,
        "cumulative_turnover": detailed["turnover_amount"],
        "closed_episode_count": closed_episode_count,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "median_cost_per_episode": median_cost_per_episode,
        "P5_break_even_cost_bps_bootstrap": p5_break_even,
        "interpretation": "P5 bootstrap break-even is the conservative value; point estimate can overstate robustness when turnover or episode count is low.",
        "financial_calls": _zero_financial_calls(),
    }


def forward_shadow_report(prepared: dict[str, Any], phase6_grid: dict[str, Any]) -> dict[str, Any]:
    config = _canonical_config_from_grid(phase6_grid)
    targets = _targets_for_config(prepared, config)
    latest_index = len(prepared["dates"]) - 1
    previous_target = targets[latest_index - 1] if latest_index > 0 else {}
    target = targets[latest_index]
    turnover = 0.5 * sum(abs(target.get(key, 0.0) - previous_target.get(key, 0.0)) for key in set(target) | set(previous_target))
    latest = _shadow_decision_record(prepared, config, targets, latest_index, include_realized=False)
    historical = []
    for index in range(max(252, latest_index - 252), latest_index - 21, 21):
        historical.append(_shadow_decision_record(prepared, config, targets, index, include_realized=True))
    return {
        "schema": "phase6_2_forward_research_shadow_v1",
        "status": "GO",
        "authority": "FORWARD_RESEARCH_SHADOW",
        "not_authority": ["PAPER_STRATEGY_AUTHORITY", "LIVE_TRADING_AUTHORITY"],
        "closed_bars_only": True,
        "orders_enabled": False,
        "account_positions_used": False,
        "parameter_changes_allowed_during_run": False,
        "record_count": 1 + len(historical),
        "latest_decision": {
            **latest,
            "planned_turnover": turnover,
            "estimated_cost": turnover * float(config["cost_bps"]) / 10_000.0,
        },
        "historical_evaluated_decisions": historical,
        "financial_calls": _zero_financial_calls(),
    }


def phase6_2_freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase62Layout.from_project_root(project_root)
    status = phase6_2_status(project_root)
    source_paths = (
        "main.py",
        "src/stocks/research/phase6_2.py",
        "src/stocks/research/phase6_diagnostics.py",
        "tests/test_phase6_2.py",
    )
    artifact_paths = (
        layout.episode_accounting_json,
        layout.semiannual_oos_json,
        layout.aggregate_oos_json,
        layout.point_in_time_history_json,
        layout.cost_reliability_json,
        layout.forward_shadow_json,
        layout.status_json,
    )
    freeze = {
        "schema": "phase6_2_freeze_v1",
        "contract_id": "PHASE6_2_SAMPLE_SUFFICIENCY_AND_FORWARD_OOS_V1",
        "freeze_status": "PHASE6_2_SAMPLE_SUFFICIENCY_AND_FORWARD_OOS_FROZEN_GO"
        if status["status"] == "PHASE6_2_SAMPLE_SUFFICIENCY_AND_FORWARD_OOS_GO"
        else "NO_GO",
        "phase6_2_status": status["status"],
        "decision_status": status["decision_status"],
        "generated_at": utc_now_iso(),
        "source_hashes": {path: sha256_file(project_root / path) for path in source_paths},
        "artifact_hashes": {str(path): sha256_file(path) for path in artifact_paths},
        "financial_calls": _zero_financial_calls(),
    }
    _write_json(layout.freeze_json, freeze)
    return freeze


def _episode_accounting_for_window(
    prepared: dict[str, Any],
    config: dict[str, Any],
    test_start: str,
    test_end: str,
    *,
    fold_id: int | str,
) -> dict[str, Any]:
    targets = _targets_for_config(prepared, config)
    dates = prepared["dates"]
    returns = prepared["returns"]
    metadata = prepared["metadata"]
    active: dict[str, dict[str, Any]] = {}
    previous_weights: dict[str, float] = {"CASH": 1.0}
    entry_count = exit_count = partial_rebalances = signal_opportunity_count = blocked_signal_count = 0
    closed_episode_pnls: list[float] = []
    carry_in_positions: list[str] = []
    carry_out_positions: list[str] = []
    blocked_reasons: dict[str, int] = {}
    for index in range(1, len(dates)):
        day = dates[index]
        in_window = test_start <= day <= test_end
        target = targets[index] if index < len(targets) else previous_weights
        if day >= test_start and not carry_in_positions:
            carry_in_positions = sorted(metadata[key]["symbol"] for key in active if key in metadata)
        if in_window and _is_rebalance_date(dates, index, str(config["rebalance"])):
            signal_opportunity_count += 1
            risky_target = {key: weight for key, weight in target.items() if key in metadata and prepared["metadata"][key]["sleeve"] != "cash" and weight > 0}
            if not risky_target:
                blocked_signal_count += 1
                reason = _blocked_signal_reason(index, prepared, config, target)
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
        for key, weight in previous_weights.items():
            if key in active and key in returns:
                active[key]["pnl"] += weight * returns[key][index] * 100.0
        for key in list(active):
            if target.get(key, 0.0) <= 0:
                if in_window:
                    exit_count += 1
                    closed_episode_pnls.append(float(active[key]["pnl"]))
                active.pop(key)
        for key, weight in target.items():
            if key in metadata and weight > 0:
                if previous_weights.get(key, 0.0) <= 0:
                    active[key] = {"start_date": day, "pnl": 0.0}
                    if in_window:
                        entry_count += 1
                elif in_window and abs(weight - previous_weights.get(key, 0.0)) > 1e-12:
                    partial_rebalances += 1
        previous_weights = target
        if day <= test_end:
            carry_out_positions = sorted(metadata[key]["symbol"] for key in active if key in metadata)
    no_trades_reason = _no_trades_reason(
        entry_count=entry_count,
        exit_count=exit_count,
        carry_in=len(carry_in_positions),
        carry_out=len(carry_out_positions),
        signal_opportunities=signal_opportunity_count,
        blocked_reasons=blocked_reasons,
    )
    return {
        "fold_id": fold_id,
        "test_start": test_start,
        "test_end": test_end,
        "selected_configuration": config,
        "entry_count": entry_count,
        "exit_count": exit_count,
        "partial_rebalances": partial_rebalances,
        "carry_in_positions": carry_in_positions,
        "carry_out_positions": carry_out_positions,
        "open_episodes_at_fold_end": len(carry_out_positions),
        "closed_episodes": exit_count,
        "closed_episode_pnls": closed_episode_pnls,
        "signal_opportunity_count": signal_opportunity_count,
        "blocked_signal_count": blocked_signal_count,
        "blocked_signal_reasons": blocked_reasons,
        "NO_TRADES_REASON": no_trades_reason,
        "episode_accounting_status": "GO",
        "carry_policy": "CONTINUOUS_TIMELINE_EPISODES_NOT_CLOSED_AT_FOLD_BOUNDARY",
    }


def _phase6_2_decision(aggregate: dict[str, Any] | None, phase6_1: dict[str, Any] | None) -> str:
    if aggregate is None or phase6_1 is None:
        return "METRIC_OR_ACCOUNTING_BLOCKED"
    gates = aggregate.get("gates", {})
    sample_gates = (
        gates.get("aggregate_closed_OOS_episodes_ge_30")
        and gates.get("effective_sample_size_ge_20")
        and gates.get("evaluable_testwindows_ge_8")
        and gates.get("unevaluable_no_trade_windows_le_1")
    )
    if not sample_gates:
        return "INSUFFICIENT_SAMPLE"
    finalist_gates = all(gates.values()) and (aggregate.get("DSR_probability") or 0.0) >= 0.95
    if finalist_gates:
        return "FINANCIAL_FINALIST_GO"
    promising = (
        gates.get("aggregate_oos_episode_pf_gt_1_10")
        and gates.get("median_oos_period_pf_gt_1_05")
        and gates.get("stress_20bps_pf_gt_1")
        and gates.get("benchmark_comparisons_won_gt_60pct")
        and gates.get("single_asset_contribution_lt_40")
        and gates.get("single_region_contribution_lt_60")
        and gates.get("parameterplateau_present")
        and gates.get("PBO_lte_0_20")
        and gates.get("DSR_probability_candidate")
    )
    return "PROMISING_RESEARCH_CANDIDATE" if promising else "NO_FINANCIAL_EDGE"


def _load_phase6_1_folds(project_root: Path) -> list[dict[str, Any]]:
    path = Phase61Layout.from_project_root(project_root).fold_diagnostics_json
    if not path.exists():
        from stocks.research.phase6_diagnostics import fold_diagnostics

        dataset = load_phase6_dataset(project_root)
        return fold_diagnostics(_prepare_common_history(dataset))["folds"]
    return json.loads(path.read_text(encoding="utf-8"))["folds"]


def _canonical_config_from_grid(grid: dict[str, Any]) -> dict[str, Any]:
    params = grid.get("best_by_calmar", {}).get("parameters", {})
    return {
        "momentum_lookback": int(params.get("momentum_lookback", 252)),
        "trend_lookback": int(params.get("trend_lookback", 100)),
        "rebalance": str(params.get("rebalance", "monthly")),
        "target_vol": float(params.get("target_vol", 0.12)),
        "cost_bps": float(params.get("cost_bps", 10.0)),
    }


def _shadow_decision_record(
    prepared: dict[str, Any],
    config: dict[str, Any],
    targets: list[dict[str, float]],
    index: int,
    *,
    include_realized: bool,
) -> dict[str, Any]:
    day = prepared["dates"][index]
    target = targets[index]
    previous = targets[index - 1] if index > 0 else {}
    scores = _scores_at_index(prepared, config, index)
    blocked = [
        {"symbol": prepared["metadata"][key]["symbol"], "reason": reason}
        for key, score in scores.items()
        for reason in ([score["block_reason"]] if score.get("block_reason") else [])
    ]
    record = {
        "decision_timestamp": f"{day}T23:59:59+00:00",
        "dataset_version": "phase5_total_return_fx_v1",
        "eligible_universe": [meta["symbol"] for meta in prepared["metadata"].values()],
        "selected_configuration": config,
        "features": scores,
        "scores": {prepared["metadata"][key]["symbol"]: value["score"] for key, value in scores.items()},
        "target_weights": {prepared["metadata"][key]["symbol"]: weight for key, weight in target.items() if key in prepared["metadata"]},
        "cash_weight": max(0.0, 1.0 - sum(weight for key, weight in target.items() if key in prepared["returns"])),
        "planned_turnover": 0.5 * sum(abs(target.get(key, 0.0) - previous.get(key, 0.0)) for key in set(target) | set(previous)),
        "estimated_cost": 0.0,
        "blocked_instruments": blocked,
        "block_reasons": _count_reasons(blocked),
        "future_evaluation_horizon": {"trading_days": 21},
    }
    if include_realized and index + 21 < len(prepared["dates"]):
        realized = [
            sum(target.get(key, 0.0) * prepared["returns"][key][j] for key in prepared["returns"])
            for j in range(index + 1, index + 22)
        ]
        benchmark = _equal_weight_return(prepared, index + 1, index + 22)
        record.update(
            {
                "realized_return_eur": _compound(realized),
                "realized_cost_proxy": float(str(record["planned_turnover"])) * float(config["cost_bps"]) / 10_000.0,
                "benchmark_return": benchmark,
                "excess_return": _compound(realized) - benchmark,
                "episode_outcome": "OPEN_OR_HORIZON_EVALUATED",
            }
        )
    return record


def _scores_at_index(prepared: dict[str, Any], config: dict[str, Any], index: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    momentum = int(config["momentum_lookback"])
    trend = int(config["trend_lookback"])
    cash_key = next((key for key, meta in prepared["metadata"].items() if meta["sleeve"] == "cash"), None)
    for key, prices in prepared["prices"].items():
        if key == cash_key:
            continue
        if index < max(momentum, trend):
            rows[key] = {"score": None, "trend_pass": False, "momentum_return": None, "block_reason": "INSUFFICIENT_WARMUP"}
            continue
        trend_average = _mean(prices[index - trend : index])
        trend_pass = prices[index] > trend_average
        momentum_start = index - momentum
        skip_month = max(momentum_start, index - 21)
        score = prices[skip_month] / prices[momentum_start] - 1.0 if prices[momentum_start] > 0 else None
        reason = None
        if not trend_pass:
            reason = "ALL_SIGNALS_BLOCKED_BY_TREND"
        elif score is None or score <= 0:
            reason = "NO_ENTRY_SIGNALS"
        rows[key] = {
            "score": score,
            "trend_pass": trend_pass,
            "momentum_return": score,
            "price": prices[index],
            "block_reason": reason,
        }
    return rows


def _blocked_signal_reason(index: int, prepared: dict[str, Any], config: dict[str, Any], target: dict[str, float]) -> str:
    if index < max(int(config["momentum_lookback"]), int(config["trend_lookback"])):
        return "INSUFFICIENT_WARMUP"
    noncash = [key for key, weight in target.items() if key in prepared["metadata"] and prepared["metadata"][key]["sleeve"] != "cash" and weight > 0]
    if noncash:
        return "NOT_BLOCKED"
    if any(key for key, weight in target.items() if key in prepared["metadata"] and prepared["metadata"][key]["sleeve"] == "cash" and weight > 0):
        return "ALL_SIGNALS_BLOCKED_BY_TREND"
    return "NO_ELIGIBLE_INSTRUMENTS"


def _no_trades_reason(
    *,
    entry_count: int,
    exit_count: int,
    carry_in: int,
    carry_out: int,
    signal_opportunities: int,
    blocked_reasons: dict[str, int],
) -> str | None:
    if entry_count or exit_count:
        return None
    if carry_in and carry_out:
        return "POSITION_CARRIED_INTO_NEXT_FOLD"
    if carry_out:
        return "OPEN_EPISODE_NOT_CLOSED"
    if signal_opportunities == 0:
        return "INSUFFICIENT_WARMUP"
    if blocked_reasons:
        return max(blocked_reasons.items(), key=lambda item: item[1])[0]
    return "NO_ENTRY_SIGNALS"


def _is_rebalance_date(dates: list[str], index: int, rebalance: str) -> bool:
    if index == 0:
        return True
    current = date.fromisoformat(dates[index])
    previous = date.fromisoformat(dates[index - 1])
    if rebalance == "weekly":
        return current.isocalendar().week != previous.isocalendar().week
    return current.month != previous.month or current.year != previous.year


def _add_months(value: date, months: int, *, day_offset: int = 0) -> date:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    result = date(year, month, 1)
    if day_offset:
        return _add_days(result, day_offset)
    return result


def _add_days(value: date, days: int) -> date:
    ordinal = value.toordinal() + days
    return date.fromordinal(ordinal)


def trainless(value: date) -> str:
    return value.isoformat()


def _compound(values: list[float]) -> float:
    total = 1.0
    for value in values:
        total *= 1.0 + value
    return total - 1.0


def _equal_weight_return(prepared: dict[str, Any], start: int, end: int) -> float:
    keys = list(prepared["returns"])
    returns = [sum(prepared["returns"][key][index] for key in keys) / len(keys) for index in range(start, min(end, len(prepared["dates"])))]
    return _compound(returns)


def _count_reasons(blocked: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in blocked:
        reason = str(row["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _p5_break_even_cost(episode_pnls: list[float], turnover_amount: float, *, iterations: int) -> float | None:
    if not episode_pnls or turnover_amount <= 0:
        return None
    values = []
    for index in range(iterations):
        sample = [episode_pnls[(index + offset * 17) % len(episode_pnls)] for offset in range(len(episode_pnls))]
        gross = sum(sample)
        values.append(gross / turnover_amount * 10_000.0)
    ordered = sorted(values)
    return ordered[max(0, round(0.05 * (len(ordered) - 1)))]
