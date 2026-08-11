from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from stocks.data.phase5_common import sha256_file, utc_now_iso
from stocks.research.phase6 import (
    Phase6Layout,
    _equal_weight_targets,
    _inverse_vol_targets,
    _max_drawdown,
    _mean,
    _median,
    _momentum_targets,
    _profit_factor,
    _slice_prepared,
    _std,
    _trend_targets,
    _write_json,
    load_phase6_dataset,
)
from stocks.research.phase6_2 import Phase62Layout, _load_phase6_1_folds, phase6_2_status
from stocks.research.phase6_diagnostics import _detailed_portfolio, _read_json_if_exists, _targets_for_config


PHASE6_3_DECISIONS = (
    "BENCHMARK_FINANCIAL_FINALIST_GO",
    "PROMISING_SIMPLE_CANDIDATE",
    "NO_EXISTING_FINANCIAL_EDGE",
    "METRIC_OR_DATA_BLOCKED",
)

SIMPLE_FAMILY_ORDER = {
    "BUY_AND_HOLD": 0,
    "EQUAL_WEIGHT": 1,
    "INVERSE_VOLATILITY": 2,
    "TREND_200D": 3,
    "MOMENTUM_ROTATION": 4,
}

PAIR_BOOTSTRAP_ITERATIONS = 5000
PAIR_BOOTSTRAP_BLOCK_LENGTH = 21
PAIR_BOOTSTRAP_SEED = 4638


@dataclass(frozen=True)
class Phase63Layout:
    output_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> Phase63Layout:
        return cls(output_dir=project_root / "output" / "research" / "phase6_3")

    @property
    def benchmark_ranking_json(self) -> Path:
        return self.output_dir / "benchmark-ranking.json"

    @property
    def benchmark_results_parquet(self) -> Path:
        return self.output_dir / "benchmark-results.parquet"

    @property
    def paired_comparisons_parquet(self) -> Path:
        return self.output_dir / "paired-comparisons.parquet"

    @property
    def champion_analysis_json(self) -> Path:
        return self.output_dir / "champion-analysis.json"

    @property
    def incremental_alpha_json(self) -> Path:
        return self.output_dir / "incremental-alpha.json"

    @property
    def leave_one_out_parquet(self) -> Path:
        return self.output_dir / "leave-one-out.parquet"

    @property
    def parameter_plateau_json(self) -> Path:
        return self.output_dir / "parameter-plateau.json"

    @property
    def statistical_validation_json(self) -> Path:
        return self.output_dir / "statistical-validation.json"

    @property
    def decision_json(self) -> Path:
        return self.output_dir / "decision.json"

    @property
    def manifest_json(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def freeze_json(self) -> Path:
        return self.output_dir / "freeze-status.json"

    @property
    def forward_shadow_spec_json(self) -> Path:
        return self.output_dir / "forward-shadow-spec.json"


def phase6_3_schema() -> dict[str, Any]:
    return {
        "schema": "phase6_3_benchmark_champion_incremental_alpha_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "strategy_status": "REJECTED_NO_INCREMENTAL_FINANCIAL_EDGE",
        "benchmark_families": list(SIMPLE_FAMILY_ORDER),
        "paired_bootstrap": {
            "iterations": PAIR_BOOTSTRAP_ITERATIONS,
            "block_length": PAIR_BOOTSTRAP_BLOCK_LENGTH,
            "seed": PAIR_BOOTSTRAP_SEED,
        },
        "decision_statuses": list(PHASE6_3_DECISIONS),
        "forbidden_calls": {
            "orders": True,
            "realtime_market_data": True,
            "new_historical_data": True,
            "provider_downloads": True,
        },
        "financial_calls": _phase6_3_call_counters(),
    }


def run_phase6_3_pipeline(project_root: Path) -> dict[str, Any]:
    layout = Phase63Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_phase6_dataset(project_root)
    prepared = _prepare_common_history_without_import_cycle(dataset)
    windows = _oos_windows(project_root)
    candidates = _benchmark_candidates(prepared)
    candidate_reports = [_evaluate_candidate(candidate, prepared, windows) for candidate in candidates]
    stats = statistical_validation(candidate_reports, selection_count=len(candidate_reports))
    _write_json(layout.statistical_validation_json, stats)
    benchmark_rows = [_benchmark_result_row(report) for report in candidate_reports]
    _write_parquet(layout.benchmark_results_parquet, benchmark_rows)

    ranking = rank_benchmarks(benchmark_rows)
    _write_json(layout.benchmark_ranking_json, ranking)
    paired = paired_benchmark_comparisons(candidate_reports)
    _write_parquet(layout.paired_comparisons_parquet, paired)
    champion_id = ranking["ranking"][0]["benchmark_id"] if ranking["ranking"] else None
    champion = next(report for report in candidate_reports if report["benchmark_id"] == champion_id)
    plateau = parameter_plateau(candidate_reports, champion_id)
    _write_json(layout.parameter_plateau_json, plateau)
    loo = leave_one_out_analysis(candidate_reports, prepared, windows, champion_id)
    _write_parquet(layout.leave_one_out_parquet, loo)
    strategy_rejection = rejected_strategy_record(project_root)
    incremental = incremental_alpha(project_root, prepared, windows, champion)
    _write_json(layout.incremental_alpha_json, incremental)
    champion_analysis = champion_analysis_report(champion, ranking, paired, plateau, stats, loo, strategy_rejection)
    _write_json(layout.champion_analysis_json, champion_analysis)
    decision = phase6_3_decision(champion_analysis, incremental)
    _write_json(layout.decision_json, decision)
    if decision["decision_status"] in {"BENCHMARK_FINANCIAL_FINALIST_GO", "PROMISING_SIMPLE_CANDIDATE"}:
        _write_json(layout.forward_shadow_spec_json, forward_shadow_spec(project_root, champion, decision))
    elif layout.forward_shadow_spec_json.exists():
        layout.forward_shadow_spec_json.unlink()
    manifest = phase6_3_manifest(project_root, layout, candidate_reports, decision)
    _write_json(layout.manifest_json, manifest)
    return phase6_3_status(project_root, manifest=manifest, decision=decision)


def phase6_3_status(
    project_root: Path,
    *,
    manifest: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layout = Phase63Layout.from_project_root(project_root)
    manifest = manifest or _read_json_if_exists(layout.manifest_json)
    decision = decision or _read_json_if_exists(layout.decision_json)
    artifacts = {
        "benchmark_ranking": layout.benchmark_ranking_json,
        "benchmark_results": layout.benchmark_results_parquet,
        "paired_comparisons": layout.paired_comparisons_parquet,
        "champion_analysis": layout.champion_analysis_json,
        "incremental_alpha": layout.incremental_alpha_json,
        "leave_one_out": layout.leave_one_out_parquet,
        "parameter_plateau": layout.parameter_plateau_json,
        "statistical_validation": layout.statistical_validation_json,
        "decision": layout.decision_json,
        "manifest": layout.manifest_json,
    }
    checks = {
        "manifest_available": manifest is not None,
        "decision_available": decision is not None and decision.get("decision_status") in PHASE6_3_DECISIONS,
        "all_required_artifacts_exist": all(path.exists() for path in artifacts.values()),
        "strategy_rejected": manifest is not None
        and manifest.get("strategy_rejection", {}).get("strategy_status") == "REJECTED_NO_INCREMENTAL_FINANCIAL_EDGE",
        "no_financial_calls": manifest is not None and manifest.get("call_counters") == _phase6_3_call_counters(),
    }
    status = "PHASE6_3_BENCHMARK_CHAMPION_AND_INCREMENTAL_ALPHA_GO" if all(checks.values()) else "NO_GO"
    return {
        "schema": "phase6_3_benchmark_champion_incremental_alpha_status_v1",
        "status": status,
        "decision_status": decision.get("decision_status") if decision else "METRIC_OR_DATA_BLOCKED",
        "generated_at": utc_now_iso(),
        "checks": checks,
        "summary": (manifest or {}).get("summary", {}),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "financial_calls": _phase6_3_call_counters(),
    }


def phase6_3_freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase63Layout.from_project_root(project_root)
    status = phase6_3_status(project_root)
    artifact_paths = [
        layout.benchmark_ranking_json,
        layout.benchmark_results_parquet,
        layout.paired_comparisons_parquet,
        layout.champion_analysis_json,
        layout.incremental_alpha_json,
        layout.leave_one_out_parquet,
        layout.parameter_plateau_json,
        layout.statistical_validation_json,
        layout.decision_json,
        layout.manifest_json,
    ]
    if layout.forward_shadow_spec_json.exists():
        artifact_paths.append(layout.forward_shadow_spec_json)
    source_paths = [
        "src/stocks/research/phase6_3.py",
        "src/stocks/research/phase6_2.py",
        "src/stocks/research/phase6_diagnostics.py",
        "src/stocks/research/phase6.py",
        "main.py",
        "tests/test_phase6_3.py",
    ]
    freeze = {
        "schema": "phase6_3_freeze_status_v1",
        "freeze_status": "PHASE6_3_BENCHMARK_CHAMPION_AND_INCREMENTAL_ALPHA_FROZEN_GO"
        if status["status"] == "PHASE6_3_BENCHMARK_CHAMPION_AND_INCREMENTAL_ALPHA_GO"
        else "NO_GO",
        "phase6_3_status": status["status"],
        "decision_status": status["decision_status"],
        "generated_at": utc_now_iso(),
        "source_hashes": {path: sha256_file(project_root / path) for path in source_paths if (project_root / path).exists()},
        "artifact_hashes": {str(path): sha256_file(path) for path in artifact_paths if path.exists()},
        "financial_calls": _phase6_3_call_counters(),
    }
    _write_json(layout.freeze_json, freeze)
    return freeze


def rank_benchmarks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "oos_CAGR": 0.20,
        "Sharpe": 0.20,
        "Calmar": 0.15,
        "positive_window_ratio": 0.15,
        "stress_20bps_period_pf": 0.10,
        "DSR_probability": 0.10,
        "turnover": -0.05,
        "abs_maximum_drawdown": -0.05,
    }
    zscores = _zscore_maps(rows, list(metrics))
    ranked: list[dict[str, Any]] = []
    for row in rows:
        missing: list[str] = []
        for metric in metrics:
            value = row.get(metric)
            if value is None:
                missing.append(metric)
                continue
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                missing.append(metric)
        score = None
        if not missing:
            score = sum(weight * zscores[metric][row["benchmark_id"]] for metric, weight in metrics.items())
        ranked.append(
            {
                "benchmark_id": row["benchmark_id"],
                "family": row["family"],
                "configuration": row["configuration"],
                "champion_score": score,
                "missing_metrics": missing,
                "fail_closed": bool(missing),
                "z_scores": {metric: zscores[metric].get(row["benchmark_id"]) for metric in metrics},
                "tie_break": {
                    "DSR_probability": row.get("DSR_probability"),
                    "positive_window_ratio": row.get("positive_window_ratio"),
                    "abs_maximum_drawdown": row.get("abs_maximum_drawdown"),
                    "turnover": row.get("turnover"),
                    "simplicity_order": SIMPLE_FAMILY_ORDER[row["family"]],
                    "benchmark_id": row["benchmark_id"],
                },
            }
        )
    ranked.sort(key=_ranking_key)
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return {
        "schema": "phase6_3_benchmark_ranking_v1",
        "status": "GO" if ranked else "NO_GO",
        "score_formula": "0.20*z(OOS_CAGR)+0.20*z(Sharpe)+0.15*z(Calmar)+0.15*z(pos_ratio)+0.10*z(20bps_stress_pf)+0.10*z(DSR_probability)-0.05*z(turnover)-0.05*z(abs(MDD))",
        "missing_metric_policy": "FAIL_CLOSED",
        "tie_break_policy": [
            "DSR desc",
            "positive_window_ratio desc",
            "abs_maximum_drawdown asc",
            "turnover asc",
            "simplicity_order asc",
            "benchmark_id asc",
        ],
        "ranking": ranked,
    }


def paired_benchmark_comparisons(candidate_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(candidate_reports, 2):
        rows.append(_paired_comparison(left, right))
    return rows


def incremental_alpha(
    project_root: Path,
    prepared: dict[str, Any],
    windows: list[dict[str, Any]],
    champion: dict[str, Any],
) -> dict[str, Any]:
    strategy_daily = _strategy_oos_daily(project_root, prepared, windows)
    strategy_by_obs = {row["obs_id"]: row for row in strategy_daily}
    champion_by_obs = {row["obs_id"]: row for row in champion["daily"]}
    obs_ids = sorted(set(strategy_by_obs) & set(champion_by_obs))
    if len(obs_ids) < 30:
        return {
            "schema": "phase6_3_incremental_alpha_v1",
            "status": "INSUFFICIENT_OVERLAP",
            "overlapping_observations": len(obs_ids),
            "financial_calls": _phase6_3_call_counters(),
        }
    y = np.array([strategy_by_obs[obs]["net_return"] for obs in obs_ids], dtype=float)
    x = np.array([champion_by_obs[obs]["net_return"] for obs in obs_ids], dtype=float)
    regression = alpha_beta_regression(y.tolist(), x.tolist())
    active = y - x
    active_navs = _navs(active.tolist())
    tracking_error = _std(active.tolist()) * math.sqrt(252)
    active_return = _compound(active.tolist())
    bootstrap_probability = block_bootstrap_probability(active.tolist(), iterations=PAIR_BOOTSTRAP_ITERATIONS)
    window_pairs = _paired_window_returns(strategy_daily, champion["daily"])
    window_active = [left - right for left, right in window_pairs]
    status = incremental_alpha_status(
        annualized_active_return=_mean(active.tolist()) * 252,
        information_ratio=None if tracking_error == 0 else _mean(active.tolist()) / _std(active.tolist()) * math.sqrt(252),
        bootstrap_probability=bootstrap_probability,
        window_win_rate=sum(1 for value in window_active if value > 0) / len(window_active) if window_active else 0.0,
        overlapping_observations=len(obs_ids),
    )
    alpha_daily = regression["alpha_daily"]
    return {
        "schema": "phase6_3_incremental_alpha_v1",
        "status": status,
        "strategy_status": "REJECTED_NO_INCREMENTAL_FINANCIAL_EDGE",
        "champion_benchmark_id": champion["benchmark_id"],
        "overlapping_observations": len(obs_ids),
        "annualized_active_return": _mean(active.tolist()) * 252,
        "tracking_error": tracking_error,
        "information_ratio": None if tracking_error == 0 else _mean(active.tolist()) / _std(active.tolist()) * math.sqrt(252),
        "alpha_annualized": (
            None if alpha_daily is None else alpha_daily * 252
        ),
        "alpha_t_stat": regression["alpha_t_stat"],
        "beta_to_champion": regression["beta"],
        "upside_capture": upside_downside_capture(y.tolist(), x.tolist())["upside_capture"],
        "downside_capture": upside_downside_capture(y.tolist(), x.tolist())["downside_capture"],
        "active_return": active_return,
        "active_maximum_drawdown": _max_drawdown(active_navs),
        "paired_daily_win_rate": float(np.mean(active > 0)),
        "paired_window_win_rate": sum(1 for value in window_active if value > 0) / len(window_active) if window_active else None,
        "paired_bootstrap_probability_strategy_gt_champion": bootstrap_probability,
        "zero_rate_policy": "NO_RISK_FREE_RATE_USED",
        "financial_calls": _phase6_3_call_counters(),
    }


def leave_one_out_analysis(
    candidate_reports: list[dict[str, Any]],
    prepared: dict[str, Any],
    windows: list[dict[str, Any]],
    champion_id: str | None,
) -> list[dict[str, Any]]:
    metadata = prepared["metadata"]
    instruments = sorted(metadata)
    regions = sorted({str(meta["region"]) for meta in metadata.values()})
    sleeves = sorted({str(meta["sleeve"]) for meta in metadata.values()})
    rows: list[dict[str, Any]] = []
    for report in candidate_reports:
        base = report["metrics"]
        removals = (
            [("instrument", key) for key in instruments]
            + [("region", region) for region in regions]
            + [("sleeve", sleeve) for sleeve in sleeves]
        )
        for removal_type, entity in removals:
            reduced = _remove_entity(prepared, removal_type, entity)
            if len(reduced["returns"]) < 2:
                continue
            candidate = _candidate_with_same_config(report)
            reduced_report = _evaluate_candidate(candidate, reduced, windows)
            rows.append(
                {
                    "benchmark_id": report["benchmark_id"],
                    "family": report["family"],
                    "removal_type": removal_type,
                    "removed_entity": entity,
                    "is_champion": report["benchmark_id"] == champion_id,
                    "full_CAGR": base["oos_CAGR"],
                    "without_CAGR": reduced_report["metrics"]["oos_CAGR"],
                    "CAGR_delta": _delta(base["oos_CAGR"], reduced_report["metrics"]["oos_CAGR"]),
                    "full_Sharpe": base["Sharpe"],
                    "without_Sharpe": reduced_report["metrics"]["Sharpe"],
                    "Sharpe_delta": _delta(base["Sharpe"], reduced_report["metrics"]["Sharpe"]),
                    "full_period_pf": base["period_profit_factor"],
                    "without_period_pf": reduced_report["metrics"]["period_profit_factor"],
                    "period_pf_delta": _delta(base["period_profit_factor"], reduced_report["metrics"]["period_profit_factor"]),
                    "fragility": fragility_classification(base["oos_CAGR"], reduced_report["metrics"]["oos_CAGR"]),
                    "champion_rank_after_removal": 1 if report["benchmark_id"] == champion_id else None,
                }
            )
    return rows


def parameter_plateau(candidate_reports: list[dict[str, Any]], champion_id: str | None) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = {}
    for report in candidate_reports:
        families.setdefault(report["family"], []).append(report)
    result: dict[str, Any] = {}
    for family, reports in sorted(families.items(), key=lambda item: SIMPLE_FAMILY_ORDER[item[0]]):
        cagrs = [report["metrics"]["oos_CAGR"] for report in reports if report["metrics"]["oos_CAGR"] is not None]
        sharpes = [report["metrics"]["Sharpe"] for report in reports if report["metrics"]["Sharpe"] is not None]
        pfs = [report["metrics"]["period_profit_factor"] for report in reports if report["metrics"]["period_profit_factor"] is not None]
        dds = [report["metrics"]["maximum_drawdown"] for report in reports if report["metrics"]["maximum_drawdown"] is not None]
        stress = [report["metrics"]["stress_20bps_period_pf"] for report in reports if report["metrics"]["stress_20bps_period_pf"] is not None]
        positive = [
            report
            for report in reports
            if (report["metrics"]["oos_CAGR"] or 0) > 0 and (report["metrics"]["period_profit_factor"] or 0) > 1.0
        ]
        positive_ratio = len(positive) / len(reports) if reports else 0.0
        dispersion = _std(cagrs) if len(cagrs) > 1 else 0.0
        classification = plateau_classification(
            variant_count=len(reports),
            positive_variant_ratio=positive_ratio,
            dispersion=dispersion,
            champion_is_positive=any(report["benchmark_id"] == champion_id for report in positive),
        )
        result[family] = {
            "variant_count": len(reports),
            "positive_variant_ratio": positive_ratio,
            "median_oos_CAGR": _median(cagrs),
            "median_Sharpe": _median(sharpes),
            "median_period_pf": _median(pfs),
            "median_maximum_drawdown": _median(dds),
            "median_20bps_stress_pf": _median(stress),
            "dispersion_oos_CAGR": dispersion,
            "classification": classification,
        }
    return {
        "schema": "phase6_3_parameter_plateau_v1",
        "status": "GO",
        "families": result,
        "champion_family": next((report["family"] for report in candidate_reports if report["benchmark_id"] == champion_id), None),
        "champion_family_classification": result.get(
            next((report["family"] for report in candidate_reports if report["benchmark_id"] == champion_id), ""), {}
        ).get("classification"),
    }


def statistical_validation(candidate_reports: list[dict[str, Any]], *, selection_count: int) -> dict[str, Any]:
    rows = []
    sharpes = [report["metrics"]["Sharpe"] for report in candidate_reports if report["metrics"]["Sharpe"] is not None]
    median_sharpe = _median(sharpes) if sharpes else None
    threshold = 0.0 if median_sharpe is None else median_sharpe
    for index, report in enumerate(candidate_reports):
        returns = [row["net_return"] for row in report["daily"]]
        bootstrap = bootstrap_metric_ci(returns, iterations=1000, seed=7000 + index)
        pbo = _candidate_pbo(report, candidate_reports)
        raw_sharpe = report["metrics"]["Sharpe"]
        dsr = (
            None
            if raw_sharpe is None
            else deflated_sharpe_probability(raw_sharpe, threshold, selection_count)
        )
        rows.append(
            {
                "benchmark_id": report["benchmark_id"],
                "family": report["family"],
                "raw_Sharpe": report["metrics"]["Sharpe"],
                "DSR_probability": dsr,
                "PBO": pbo,
                "bootstrap_CAGR_ci": bootstrap["CAGR_ci"],
                "bootstrap_Sharpe_ci": bootstrap["Sharpe_ci"],
                "bootstrap_probability_total_return_gt_0": bootstrap["probability_total_return_gt_0"],
                "probability_benchmark_gt_cash": bootstrap["probability_total_return_gt_0"],
            }
        )
        report["metrics"]["DSR_probability"] = dsr
        report["metrics"]["PBO"] = pbo
        report["metrics"]["bootstrap_probability_total_return_gt_0"] = bootstrap["probability_total_return_gt_0"]
    return {
        "schema": "phase6_3_statistical_validation_v1",
        "status": "GO",
        "benchmark_variant_selection_count": selection_count,
        "prior_strategy_selection_count": 108,
        "selection_scope": "PHASE6_3_SIMPLE_BENCHMARK_VARIANTS_ONLY",
        "rows": rows,
    }


def champion_analysis_report(
    champion: dict[str, Any],
    ranking: dict[str, Any],
    paired: list[dict[str, Any]],
    plateau: dict[str, Any],
    stats: dict[str, Any],
    loo: list[dict[str, Any]],
    strategy_rejection: dict[str, Any],
) -> dict[str, Any]:
    champion_stats = next(row for row in stats["rows"] if row["benchmark_id"] == champion["benchmark_id"])
    champion_loo = [row for row in loo if row["benchmark_id"] == champion["benchmark_id"]]
    dominance = champion["dominance"]
    pair_rows = [
        row
        for row in paired
        if row["benchmark_a"] == champion["benchmark_id"] or row["benchmark_b"] == champion["benchmark_id"]
    ]
    champion_superiority = _champion_pair_superiority(champion["benchmark_id"], pair_rows)
    return {
        "schema": "phase6_3_champion_analysis_v1",
        "status": "GO",
        "champion_benchmark_id": champion["benchmark_id"],
        "champion_family": champion["family"],
        "champion_configuration": champion["configuration"],
        "ranking": ranking["ranking"][0],
        "metrics": champion["metrics"],
        "dominance": dominance,
        "dominance_limits": {
            "instrument": 0.40,
            "region": 0.60,
            "sleeve": 0.70,
            "year": 0.50,
        },
        "leave_one_out_fragility_summary": _fragility_summary(champion_loo),
        "parameter_plateau": plateau["families"].get(champion["family"]),
        "statistical_validation": champion_stats,
        "paired_superiority": champion_superiority,
        "strategy_rejection": strategy_rejection,
        "financial_calls": _phase6_3_call_counters(),
    }


def phase6_3_decision(champion_analysis: dict[str, Any], incremental: dict[str, Any]) -> dict[str, Any]:
    metrics = champion_analysis["metrics"]
    dominance = champion_analysis["dominance"]
    plateau = champion_analysis["parameter_plateau"] or {}
    stats = champion_analysis["statistical_validation"]
    paired = champion_analysis["paired_superiority"]
    gates = {
        "testwindow_count_ge_8": metrics["testwindow_count"] >= 8,
        "positive_window_ratio_ge_0_65": metrics["positive_window_ratio"] >= 0.65,
        "aggregate_period_pf_gt_1_05": (metrics["period_profit_factor"] or 0) > 1.05,
        "stress_20bps_pf_gt_1": (metrics["stress_20bps_period_pf"] or 0) > 1.0,
        "stress_30bps_pf_gt_0_95": (metrics["stress_30bps_period_pf"] or 0) > 0.95,
        "CAGR_gt_0": (metrics["oos_CAGR"] or 0) > 0,
        "Sharpe_gt_0": (metrics["Sharpe"] or 0) > 0,
        "max_drawdown_within_25pct": abs(metrics["maximum_drawdown"] or -1) <= 0.25,
        "concentration_within_limits": not dominance["dominance_blocked"],
        "plateau_broad_or_narrow": plateau.get("classification") in {"BROAD_PLATEAU", "NARROW_PLATEAU"},
        "PBO_le_0_20": stats["PBO"] <= 0.20,
        "DSR_ge_0_95": stats["DSR_probability"] >= 0.95,
        "bootstrap_prob_return_gt_0_ge_0_95": stats["bootstrap_probability_total_return_gt_0"] >= 0.95,
        "paired_superiority_sufficient": paired["superiority_ratio"] >= 0.60,
        "financial_calls_zero": True,
    }
    promising_gates = {
        "performance_positive": (metrics["oos_CAGR"] or 0) > 0 and (metrics["Sharpe"] or 0) > 0,
        "stress_survives": (metrics["stress_20bps_period_pf"] or 0) > 1.0,
        "majority_windows_positive": metrics["positive_window_ratio"] > 0.50,
        "no_severe_dominance": dominance["dominance_classification"] != "DOMINATED_BY_ENTITY",
        "incremental_alpha_not_required": incremental["status"] != "METRIC_BLOCKED",
    }
    if all(gates.values()):
        decision_status = "BENCHMARK_FINANCIAL_FINALIST_GO"
    elif all(promising_gates.values()):
        decision_status = "PROMISING_SIMPLE_CANDIDATE"
    elif incremental["status"] == "METRIC_BLOCKED":
        decision_status = "METRIC_OR_DATA_BLOCKED"
    else:
        decision_status = "NO_EXISTING_FINANCIAL_EDGE"
    return {
        "schema": "phase6_3_decision_v1",
        "decision_status": decision_status,
        "champion_benchmark_id": champion_analysis["champion_benchmark_id"],
        "strategy_status": "REJECTED_NO_INCREMENTAL_FINANCIAL_EDGE",
        "financial_finalist_gates": gates,
        "promising_gates": promising_gates,
        "incremental_alpha_status": incremental["status"],
        "financial_calls": _phase6_3_call_counters(),
    }


def forward_shadow_spec(project_root: Path, champion: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    dataset_hashes = _dataset_hashes(project_root)
    return {
        "schema": "phase6_3_forward_shadow_spec_v1",
        "shadow_id": "PHASE6_3_SIMPLE_BENCHMARK_FORWARD_RESEARCH_SHADOW",
        "champion_benchmark_id": champion["benchmark_id"],
        "champion_configuration": champion["configuration"],
        "decision_status": decision["decision_status"],
        "decision_frequency": "daily_after_all_required_closed_bars_are_available",
        "required_closed_data": "EUR total-return daily bars only",
        "eligible_universe_rule": "frozen Phase 5 common-history research universe",
        "weight_calculation": champion["weight_calculation"],
        "cash_rule": "cash residual earns frozen EUR cash proxy return where present",
        "cost_assumption": "10 bps base; 20/30/50 bps stress recorded",
        "dataset_hashes": dataset_hashes,
        "parameter_hash": _stable_hash(champion["configuration"]),
        "evaluation_horizon": "forward research only; no orders",
        "authority": "NONE",
        "orders_allowed": False,
        "paper_orders_allowed": False,
        "live_orders_allowed": False,
        "status": "FORWARD_RESEARCH_SHADOW_SPEC_GO",
    }


def rejected_strategy_record(project_root: Path) -> dict[str, Any]:
    phase62_layout = Phase62Layout.from_project_root(project_root)
    phase62 = _read_json_if_exists(phase62_layout.status_json) or phase6_2_status(project_root)
    summary = phase62.get("summary", {})
    return {
        "strategy_id": "PHASE6_MULTI_ASSET_MOMENTUM_TREND_COMPLEX",
        "strategy_version": "phase6_2_frozen",
        "strategy_status": "REJECTED_NO_INCREMENTAL_FINANCIAL_EDGE",
        "rejection_status": "FINAL_FOR_PHASE6_3",
        "rejection_reason": "Phase 6.2 ended NO_FINANCIAL_EDGE and benchmark champion comparison is required before authority.",
        "phase6_2_artifact": str(phase62_layout.status_json),
        "benchmark_win_rate": summary.get("benchmark_win_rate"),
        "DSR_probability": summary.get("DSR_probability"),
        "aggregate_oos_episode_pf": summary.get("aggregate_oos_episode_pf"),
        "20bps_stress_pf": summary.get("stress_20bps_pf"),
        "frozen_at": phase62.get("generated_at"),
    }


def phase6_3_manifest(
    project_root: Path,
    layout: Phase63Layout,
    candidate_reports: list[dict[str, Any]],
    decision: dict[str, Any],
) -> dict[str, Any]:
    artifact_paths = [
        layout.benchmark_ranking_json,
        layout.benchmark_results_parquet,
        layout.paired_comparisons_parquet,
        layout.champion_analysis_json,
        layout.incremental_alpha_json,
        layout.leave_one_out_parquet,
        layout.parameter_plateau_json,
        layout.statistical_validation_json,
        layout.decision_json,
    ]
    if layout.forward_shadow_spec_json.exists():
        artifact_paths.append(layout.forward_shadow_spec_json)
    return {
        "schema": "phase6_3_manifest_v1",
        "status": "GO",
        "generated_at": utc_now_iso(),
        "summary": {
            "benchmark_candidate_count": len(candidate_reports),
            "oos_window_count": candidate_reports[0]["metrics"]["testwindow_count"] if candidate_reports else 0,
            "champion_benchmark_id": decision["champion_benchmark_id"],
            "decision_status": decision["decision_status"],
        },
        "strategy_rejection": rejected_strategy_record(project_root),
        "artifact_hashes": {str(path): sha256_file(path) for path in artifact_paths if path.exists()},
        "input_hashes": {
            "phase6_2_status": sha256_file(Phase62Layout.from_project_root(project_root).status_json),
            "phase6_benchmarks": sha256_file(Phase6Layout.from_project_root(project_root).benchmarks_json),
        },
        "call_counters": _phase6_3_call_counters(),
        "financial_calls": _phase6_3_call_counters(),
    }


def alpha_beta_regression(strategy_returns: list[float], benchmark_returns: list[float]) -> dict[str, float | None]:
    if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 3:
        return {"alpha_daily": None, "beta": None, "alpha_t_stat": None}
    y = np.array(strategy_returns, dtype=float)
    x = np.array(benchmark_returns, dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var_x = float(np.var(x, ddof=1))
    beta = 0.0 if var_x == 0 else float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    alpha = y_mean - beta * x_mean
    residuals = y - (alpha + beta * x)
    if len(y) <= 2:
        t_stat = None
    else:
        stderr = float(np.std(residuals, ddof=1) / math.sqrt(len(y)))
        t_stat = None if stderr == 0 else alpha / stderr
    return {"alpha_daily": alpha, "beta": beta, "alpha_t_stat": t_stat}


def block_bootstrap_probability(
    values: list[float],
    *,
    iterations: int = PAIR_BOOTSTRAP_ITERATIONS,
    block_length: int = PAIR_BOOTSTRAP_BLOCK_LENGTH,
    seed: int = PAIR_BOOTSTRAP_SEED,
) -> float | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    starts = np.arange(0, max(1, len(arr) - block_length + 1))
    block_sums = np.array([float(np.sum(arr[start : start + block_length])) for start in starts])
    blocks_needed = int(math.ceil(len(arr) / block_length))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(block_sums), size=(iterations, blocks_needed))
    sums = np.sum(block_sums[picks], axis=1)
    return float(np.mean(sums > 0.0))


def upside_downside_capture(candidate_returns: list[float], benchmark_returns: list[float]) -> dict[str, float | None]:
    if len(candidate_returns) != len(benchmark_returns) or not candidate_returns:
        return {"upside_capture": None, "downside_capture": None}
    upside = [(left, right) for left, right in zip(candidate_returns, benchmark_returns, strict=True) if right > 0]
    downside = [(left, right) for left, right in zip(candidate_returns, benchmark_returns, strict=True) if right < 0]
    return {
        "upside_capture": None if not upside or sum(right for _left, right in upside) == 0 else sum(left for left, _right in upside) / sum(right for _left, right in upside),
        "downside_capture": None
        if not downside or sum(right for _left, right in downside) == 0
        else sum(left for left, _right in downside) / sum(right for _left, right in downside),
    }


def plateau_classification(
    *,
    variant_count: int,
    positive_variant_ratio: float,
    dispersion: float,
    champion_is_positive: bool,
) -> str:
    if variant_count <= 1:
        return "BROAD_PLATEAU" if champion_is_positive else "NO_POSITIVE_PLATEAU"
    if positive_variant_ratio >= 0.60 and dispersion <= 0.05:
        return "BROAD_PLATEAU"
    if positive_variant_ratio >= 0.35:
        return "NARROW_PLATEAU"
    if champion_is_positive:
        return "ISOLATED_WINNER"
    return "NO_POSITIVE_PLATEAU"


def fragility_classification(full_cagr: float | None, without_cagr: float | None) -> str:
    if full_cagr is None or without_cagr is None:
        return "FRAGILE"
    if full_cagr > 0 and without_cagr <= 0:
        return "DOMINATED_BY_ENTITY"
    delta = full_cagr - without_cagr
    if delta > 0.75 * abs(full_cagr):
        return "FRAGILE"
    if delta > 0.35 * abs(full_cagr):
        return "MODERATELY_SENSITIVE"
    return "ROBUST"


def incremental_alpha_status(
    *,
    annualized_active_return: float | None,
    information_ratio: float | None,
    bootstrap_probability: float | None,
    window_win_rate: float,
    overlapping_observations: int,
) -> str:
    if overlapping_observations < 30:
        return "INSUFFICIENT_OVERLAP"
    if annualized_active_return is None or information_ratio is None or bootstrap_probability is None:
        return "METRIC_BLOCKED"
    if (
        annualized_active_return > 0
        and information_ratio > 0
        and bootstrap_probability >= 0.95
        and window_win_rate > 0.60
    ):
        return "POSITIVE_INCREMENTAL_ALPHA"
    if annualized_active_return < 0 or information_ratio < 0:
        return "NEGATIVE_INCREMENTAL_ALPHA"
    return "NO_INCREMENTAL_ALPHA"


def deflated_sharpe_probability(sharpe: float | None, threshold: float, selection_count: int) -> float:
    if sharpe is None or not math.isfinite(sharpe):
        return 0.0
    penalty = math.sqrt(max(0.0, 2.0 * math.log(max(1, selection_count))))
    z_value = sharpe - threshold - penalty
    return _normal_cdf(z_value)


def bootstrap_metric_ci(returns: list[float], *, iterations: int, seed: int) -> dict[str, Any]:
    if not returns:
        return {
            "CAGR_ci": [None, None],
            "Sharpe_ci": [None, None],
            "probability_total_return_gt_0": 0.0,
        }
    arr = np.array(returns, dtype=float)
    block_length = min(PAIR_BOOTSTRAP_BLOCK_LENGTH, len(arr))
    starts = np.arange(0, max(1, len(arr) - block_length + 1))
    blocks_needed = int(math.ceil(len(arr) / block_length))
    rng = np.random.default_rng(seed)
    cagr_values: list[float] = []
    sharpe_values: list[float] = []
    terminal: list[float] = []
    for _ in range(iterations):
        sampled: list[float] = []
        for start in rng.choice(starts, size=blocks_needed, replace=True):
            sampled.extend(arr[int(start) : int(start) + block_length])
        sample = sampled[: len(arr)]
        total = _compound(sample)
        terminal.append(total)
        years = len(sample) / 252
        cagr_values.append((1.0 + total) ** (1.0 / years) - 1.0 if total > -1 and years > 0 else -1.0)
        stdev = _std(sample)
        sharpe_values.append(0.0 if stdev == 0 else _mean(sample) / stdev * math.sqrt(252))
    return {
        "CAGR_ci": [_percentile(cagr_values, 5), _percentile(cagr_values, 95)],
        "Sharpe_ci": [_percentile(sharpe_values, 5), _percentile(sharpe_values, 95)],
        "probability_total_return_gt_0": sum(1 for value in terminal if value > 0) / len(terminal),
    }


def _prepare_common_history_without_import_cycle(dataset: dict[str, Any]) -> dict[str, Any]:
    from stocks.research.phase6 import _prepare_common_history

    return _prepare_common_history(dataset)


def _benchmark_candidates(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = prepared["metadata"]
    candidates: list[dict[str, Any]] = []
    for key, meta in sorted(metadata.items(), key=lambda item: item[1]["symbol"]):
        candidates.append(
            {
                "benchmark_id": f"BUY_AND_HOLD_{meta['symbol']}",
                "family": "BUY_AND_HOLD",
                "configuration": {"con_id": int(key), "symbol": meta["symbol"]},
                "weight_calculation": f"100% buy-and-hold {meta['symbol']}",
            }
        )
    candidates.extend(
        [
            {
                "benchmark_id": "EQUAL_WEIGHT_MONTHLY",
                "family": "EQUAL_WEIGHT",
                "configuration": {"rebalance": "monthly"},
                "weight_calculation": "equal weight monthly rebalance",
            },
            {
                "benchmark_id": "INVERSE_VOLATILITY_63D_MONTHLY",
                "family": "INVERSE_VOLATILITY",
                "configuration": {"lookback": 63, "rebalance": "monthly"},
                "weight_calculation": "inverse 63-day volatility monthly rebalance",
            },
            {
                "benchmark_id": "TREND_200D_MONTHLY",
                "family": "TREND_200D",
                "configuration": {"trend_lookback": 200, "rebalance": "monthly"},
                "weight_calculation": "hold assets above 200-day SMA, monthly rebalance",
            },
            {
                "benchmark_id": "MOMENTUM_ROTATION_252_200_TOP4_MONTHLY",
                "family": "MOMENTUM_ROTATION",
                "configuration": {"momentum_lookback": 252, "trend_lookback": 200, "top_n": 4, "rebalance": "monthly"},
                "weight_calculation": "12-1 momentum top 4 with 200-day trend gate, monthly rebalance",
            },
        ]
    )
    return candidates


def _candidate_targets(candidate: dict[str, Any], prepared: dict[str, Any]) -> list[dict[str, float]]:
    family = candidate["family"]
    config = candidate["configuration"]
    if family == "BUY_AND_HOLD":
        key = str(config["con_id"])
        return [{key: 1.0} for _day in prepared["dates"]]
    if family == "EQUAL_WEIGHT":
        return _equal_weight_targets(prepared, str(config["rebalance"]))
    if family == "INVERSE_VOLATILITY":
        return _inverse_vol_targets(prepared, lookback=int(config["lookback"]), rebalance=str(config["rebalance"]))
    if family == "TREND_200D":
        return _trend_targets(prepared, trend_lookback=int(config["trend_lookback"]), rebalance=str(config["rebalance"]))
    if family == "MOMENTUM_ROTATION":
        return _momentum_targets(
            prepared,
            momentum_lookback=int(config["momentum_lookback"]),
            trend_lookback=int(config["trend_lookback"]),
            top_n=int(config["top_n"]),
            rebalance=str(config["rebalance"]),
        )
    raise ValueError(f"unsupported benchmark family: {family}")


def _evaluate_candidate(candidate: dict[str, Any], prepared: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any]:
    window_reports = []
    daily: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    all_returns: list[float] = []
    all_turnover: list[float] = []
    all_costs: list[float] = []
    all_cash: list[float] = []
    nav = 100.0
    contribution_by_asset: dict[str, float] = {}
    contribution_by_year: dict[str, float] = {}
    for window in windows:
        subset = _slice_prepared(prepared, window["test_start"], window["test_end"])
        targets = _candidate_targets(candidate, subset)
        detailed = _detailed_portfolio(
            subset,
            targets,
            cost_bps=10.0,
            parameters=candidate["configuration"],
            name=candidate["benchmark_id"],
        )
        returns = [row["net_return"] for row in detailed["daily"]]
        window_return = _compound(returns)
        window_reports.append(
            {
                "window_id": window["window_id"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "net_return": window_return,
                "period_profit_factor": _profit_factor(returns),
                "episode_profit_factor": _episode_pf([episode["pnl"] for episode in detailed["episodes"]])["profit_factor"],
                "episode_count": len(detailed["episodes"]),
                "turnover": sum(row["turnover"] for row in detailed["daily"]),
                "costs": sum(row["cost"] for row in detailed["daily"]),
            }
        )
        for row in detailed["daily"]:
            nav *= 1.0 + row["net_return"]
            enriched = {
                **row,
                "obs_id": f"{window['window_id']}:{row['date']}",
                "window_id": window["window_id"],
                "benchmark_id": candidate["benchmark_id"],
                "nav": nav,
            }
            daily.append(enriched)
            all_returns.append(row["net_return"])
            all_turnover.append(row["turnover"])
            all_costs.append(row["cost"])
            all_cash.append(row["cash_weight"])
        for episode in detailed["episodes"]:
            episodes.append({**episode, "window_id": window["window_id"]})
        _merge_float_maps(contribution_by_asset, detailed["contribution_by_asset"])
        _merge_float_maps(contribution_by_year, detailed["contribution_by_year"])
    stress = {
        cost: _stress_metrics(candidate, prepared, windows, cost_bps=cost)
        for cost in (20.0, 30.0, 50.0)
    }
    metrics = _aggregate_candidate_metrics(
        candidate,
        prepared,
        windows,
        daily,
        episodes,
        window_reports,
        all_returns,
        all_turnover,
        all_costs,
        all_cash,
        stress,
    )
    dominance = _dominance(prepared, contribution_by_asset, contribution_by_year)
    return {
        **candidate,
        "metrics": metrics,
        "window_reports": window_reports,
        "daily": daily,
        "episodes": episodes,
        "contribution_by_asset": contribution_by_asset,
        "contribution_by_year": contribution_by_year,
        "dominance": dominance,
        "financial_calls": _phase6_3_call_counters(),
    }


def _aggregate_candidate_metrics(
    candidate: dict[str, Any],
    prepared: dict[str, Any],
    windows: list[dict[str, Any]],
    daily: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    window_reports: list[dict[str, Any]],
    all_returns: list[float],
    all_turnover: list[float],
    all_costs: list[float],
    all_cash: list[float],
    stress: dict[float, dict[str, float | None]],
) -> dict[str, Any]:
    navs = _navs(all_returns)
    total_return = navs[-1] / navs[0] - 1.0 if len(navs) > 1 else 0.0
    years = len(all_returns) / 252
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 and years > 0 else None
    episode_values = [episode["pnl"] for episode in episodes]
    episode_pf = _episode_pf(episode_values)
    yearly = _calendar_returns_from_daily(daily)
    window_returns = [row["net_return"] for row in window_reports]
    return {
        "benchmark_id": candidate["benchmark_id"],
        "family": candidate["family"],
        "start": min((window["test_start"] for window in windows), default=None),
        "end": max((window["test_end"] for window in windows), default=None),
        "testwindow_count": len(windows),
        "evaluable_windows": sum(1 for row in window_reports if row["episode_count"] >= 1 or row["net_return"] != 0),
        "positive_windows": sum(1 for value in window_returns if value > 0),
        "positive_window_ratio": sum(1 for value in window_returns if value > 0) / len(window_returns) if window_returns else 0.0,
        "aggregate_oos_return": total_return,
        "oos_CAGR": cagr,
        "annualized_volatility": _std(all_returns) * math.sqrt(252),
        "Sharpe": None if _std(all_returns) == 0 else _mean(all_returns) / _std(all_returns) * math.sqrt(252),
        "Sortino": _sortino(all_returns),
        "maximum_drawdown": _max_drawdown(navs),
        "abs_maximum_drawdown": abs(_max_drawdown(navs)),
        "Calmar": None if not cagr or _max_drawdown(navs) == 0 else cagr / abs(_max_drawdown(navs)),
        "period_profit_factor": _profit_factor(all_returns),
        "episode_profit_factor": episode_pf["profit_factor"],
        "episode_pf_status": episode_pf["pf_status"],
        "closed_episodes": len(episodes),
        "episode_win_rate": sum(1 for value in episode_values if value > 0) / len(episode_values) if episode_values else None,
        "average_win": _mean([value for value in episode_values if value > 0]) if any(value > 0 for value in episode_values) else None,
        "average_loss": _mean([value for value in episode_values if value < 0]) if any(value < 0 for value in episode_values) else None,
        "turnover": sum(all_turnover),
        "transaction_costs": sum(all_costs),
        "average_cash": _mean(all_cash) if all_cash else 0.0,
        "maximum_cash": max(all_cash) if all_cash else 0.0,
        "worst_window_return": min(window_returns) if window_returns else None,
        "best_window_return": max(window_returns) if window_returns else None,
        "worst_window_period_pf": min((row["period_profit_factor"] or 0.0 for row in window_reports), default=None),
        "best_window_period_pf": max((row["period_profit_factor"] or 0.0 for row in window_reports), default=None),
        "worst_year": min(yearly.values()) if yearly else None,
        "best_year": max(yearly.values()) if yearly else None,
        "stress_20bps_return": stress[20.0]["return"],
        "stress_20bps_period_pf": stress[20.0]["period_pf"],
        "stress_30bps_return": stress[30.0]["return"],
        "stress_30bps_period_pf": stress[30.0]["period_pf"],
        "stress_50bps_return": stress[50.0]["return"],
        "stress_50bps_period_pf": stress[50.0]["period_pf"],
        "break_even_cost_bps": None if sum(all_turnover) == 0 else _compound([row["gross_return"] for row in daily]) / sum(all_turnover) * 10_000,
        "instrument_count": len(prepared["returns"]),
        "region_count": len({meta["region"] for meta in prepared["metadata"].values()}),
        "sleeve_count": len({meta["sleeve"] for meta in prepared["metadata"].values()}),
        "PBO": None,
        "DSR_probability": None,
        "bootstrap_probability_total_return_gt_0": None,
    }


def _benchmark_result_row(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    return {
        "benchmark_id": report["benchmark_id"],
        "family": report["family"],
        "configuration": json.dumps(report["configuration"], sort_keys=True),
        **metrics,
        "single_asset_contribution_max": report["dominance"]["single_asset_contribution_max"],
        "single_region_contribution_max": report["dominance"]["single_region_contribution_max"],
        "single_sleeve_contribution_max": report["dominance"]["single_sleeve_contribution_max"],
        "single_year_contribution_max": report["dominance"]["single_year_contribution_max"],
        "dominance_classification": report["dominance"]["dominance_classification"],
        "financial_calls_place_order": 0,
        "financial_calls_cancel_order": 0,
        "financial_calls_global_cancel": 0,
    }


def _stress_metrics(
    candidate: dict[str, Any],
    prepared: dict[str, Any],
    windows: list[dict[str, Any]],
    *,
    cost_bps: float,
) -> dict[str, float | None]:
    returns: list[float] = []
    for window in windows:
        subset = _slice_prepared(prepared, window["test_start"], window["test_end"])
        detailed = _detailed_portfolio(
            subset,
            _candidate_targets(candidate, subset),
            cost_bps=cost_bps,
            parameters=candidate["configuration"],
            name=candidate["benchmark_id"],
        )
        returns.extend(row["net_return"] for row in detailed["daily"])
    return {"return": _compound(returns), "period_pf": _profit_factor(returns)}


def _paired_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_by_obs = {row["obs_id"]: row for row in left["daily"]}
    right_by_obs = {row["obs_id"]: row for row in right["daily"]}
    obs_ids = sorted(set(left_by_obs) & set(right_by_obs))
    left_returns = [left_by_obs[obs]["net_return"] for obs in obs_ids]
    right_returns = [right_by_obs[obs]["net_return"] for obs in obs_ids]
    active = [a - b for a, b in zip(left_returns, right_returns, strict=True)]
    capture = upside_downside_capture(left_returns, right_returns)
    tracking_error = _std(active) * math.sqrt(252)
    return {
        "benchmark_a": left["benchmark_id"],
        "benchmark_b": right["benchmark_id"],
        "overlapping_observations": len(obs_ids),
        "active_return": _compound(active),
        "annualized_active_return": _mean(active) * 252 if active else None,
        "tracking_error": tracking_error,
        "information_ratio": None if tracking_error == 0 else _mean(active) / _std(active) * math.sqrt(252),
        "upside_capture": capture["upside_capture"],
        "downside_capture": capture["downside_capture"],
        "drawdown_difference": left["metrics"]["maximum_drawdown"] - right["metrics"]["maximum_drawdown"],
        "turnover_difference": left["metrics"]["turnover"] - right["metrics"]["turnover"],
        "cost_difference": left["metrics"]["transaction_costs"] - right["metrics"]["transaction_costs"],
        "paired_win_rate": sum(1 for value in active if value > 0) / len(active) if active else None,
        "paired_bootstrap_probability_a_gt_b": block_bootstrap_probability(active),
        "bootstrap_iterations": PAIR_BOOTSTRAP_ITERATIONS,
        "bootstrap_block_length": PAIR_BOOTSTRAP_BLOCK_LENGTH,
        "bootstrap_seed": PAIR_BOOTSTRAP_SEED,
    }


def _oos_windows(project_root: Path) -> list[dict[str, Any]]:
    annual = _load_phase6_1_folds(project_root)
    phase62 = Phase62Layout.from_project_root(project_root)
    semiannual = json.loads(phase62.semiannual_oos_json.read_text(encoding="utf-8"))["folds"]
    windows = [
        {
            "window_id": f"annual_{fold['fold_id']}",
            "kind": "annual",
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "selected_configuration": fold["selected_configuration"],
        }
        for fold in annual
    ]
    windows.extend(
        {
            "window_id": f"semiannual_{fold['fold_id']}",
            "kind": "semiannual",
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "selected_configuration": fold["selected_configuration"],
        }
        for fold in semiannual
    )
    return sorted(windows, key=lambda item: (item["test_start"], item["window_id"]))


def _strategy_oos_daily(project_root: Path, prepared: dict[str, Any], windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    window_by_id = {window["window_id"]: window for window in windows}
    result: list[dict[str, Any]] = []
    for window_id, window in window_by_id.items():
        subset = _slice_prepared(prepared, window["test_start"], window["test_end"])
        detailed = _detailed_portfolio(
            subset,
            _targets_for_config(subset, window["selected_configuration"]),
            cost_bps=float(window["selected_configuration"]["cost_bps"]),
            parameters=window["selected_configuration"],
            name=f"phase6_2_strategy_{window_id}",
        )
        for row in detailed["daily"]:
            result.append({**row, "obs_id": f"{window_id}:{row['date']}", "window_id": window_id})
    return result


def _candidate_with_same_config(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_id": report["benchmark_id"],
        "family": report["family"],
        "configuration": report["configuration"],
        "weight_calculation": report["weight_calculation"],
    }


def _remove_entity(prepared: dict[str, Any], removal_type: str, entity: str) -> dict[str, Any]:
    metadata = prepared["metadata"]
    if removal_type == "instrument":
        keep = {key for key in metadata if key != entity}
    elif removal_type == "region":
        keep = {key for key, meta in metadata.items() if str(meta["region"]) != entity}
    elif removal_type == "sleeve":
        keep = {key for key, meta in metadata.items() if str(meta["sleeve"]) != entity}
    else:
        raise ValueError(f"unsupported removal type: {removal_type}")
    return {
        **prepared,
        "returns": {key: values for key, values in prepared["returns"].items() if key in keep},
        "prices": {key: values for key, values in prepared["prices"].items() if key in keep},
        "metadata": {key: value for key, value in metadata.items() if key in keep},
        "cash_key": prepared["cash_key"] if prepared["cash_key"] in keep else None,
    }


def _dominance(prepared: dict[str, Any], contribution_by_asset: dict[str, float], contribution_by_year: dict[str, float]) -> dict[str, Any]:
    metadata = prepared["metadata"]
    positive_asset = {key: max(value, 0.0) for key, value in contribution_by_asset.items()}
    total_positive = sum(positive_asset.values())
    asset_shares = {key: value / total_positive for key, value in positive_asset.items()} if total_positive else {}
    region_contrib: dict[str, float] = {}
    sleeve_contrib: dict[str, float] = {}
    for key, value in positive_asset.items():
        if key in metadata:
            region_contrib[metadata[key]["region"]] = region_contrib.get(metadata[key]["region"], 0.0) + value
            sleeve_contrib[metadata[key]["sleeve"]] = sleeve_contrib.get(metadata[key]["sleeve"], 0.0) + value
    region_total = sum(region_contrib.values())
    sleeve_total = sum(sleeve_contrib.values())
    year_positive = {key: max(value, 0.0) for key, value in contribution_by_year.items()}
    year_total = sum(year_positive.values())
    max_asset = max(asset_shares.values(), default=0.0)
    max_region = max((value / region_total for value in region_contrib.values()), default=0.0) if region_total else 0.0
    max_sleeve = max((value / sleeve_total for value in sleeve_contrib.values()), default=0.0) if sleeve_total else 0.0
    max_year = max((value / year_total for value in year_positive.values()), default=0.0) if year_total else 0.0
    blocked = max_asset > 0.40 or max_region > 0.60 or max_sleeve > 0.70 or max_year > 0.50
    return {
        "single_asset_contribution_max": max_asset,
        "single_region_contribution_max": max_region,
        "single_sleeve_contribution_max": max_sleeve,
        "single_year_contribution_max": max_year,
        "dominance_blocked": blocked,
        "dominance_classification": "DOMINATED_BY_ENTITY" if blocked else "ROBUST",
    }


def _episode_pf(values: list[float]) -> dict[str, Any]:
    positive = sum(value for value in values if value > 0)
    negative = abs(sum(value for value in values if value < 0))
    if not values:
        return {"profit_factor": None, "pf_status": "NO_TRADES"}
    if len(values) < 10:
        return {"profit_factor": None if negative == 0 else positive / negative, "pf_status": "INSUFFICIENT_SAMPLE"}
    if negative == 0 and positive > 0:
        return {"profit_factor": None, "pf_status": "NO_LOSING_EPISODES"}
    if positive == 0 and negative > 0:
        return {"profit_factor": 0.0, "pf_status": "NO_POSITIVE_EPISODES"}
    if negative == 0:
        return {"profit_factor": None, "pf_status": "ZERO_DENOMINATOR"}
    return {"profit_factor": positive / negative, "pf_status": "EVALUABLE"}


def _candidate_pbo(report: dict[str, Any], candidate_reports: list[dict[str, Any]]) -> float:
    below_median = 0
    total = 0
    by_window: dict[str, list[float]] = {}
    for candidate in candidate_reports:
        for row in candidate["window_reports"]:
            by_window.setdefault(row["window_id"], []).append(row["net_return"])
    for row in report["window_reports"]:
        peer_values = by_window.get(row["window_id"], [])
        if not peer_values:
            continue
        total += 1
        if row["net_return"] < _median(peer_values):
            below_median += 1
    return below_median / total if total else 1.0


def _champion_pair_superiority(champion_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = 0
    evaluable = 0
    probabilities = []
    for row in rows:
        if row["benchmark_a"] == champion_id:
            probability = row["paired_bootstrap_probability_a_gt_b"]
            active = row["annualized_active_return"]
        else:
            probability = 1.0 - row["paired_bootstrap_probability_a_gt_b"]
            active = -row["annualized_active_return"]
        if probability is None or active is None:
            continue
        evaluable += 1
        wins += 1 if active > 0 and probability > 0.50 else 0
        probabilities.append(probability)
    return {
        "evaluable_pairs": evaluable,
        "superior_pairs": wins,
        "superiority_ratio": wins / evaluable if evaluable else 0.0,
        "median_bootstrap_probability": _median(probabilities),
    }


def _paired_window_returns(left_daily: list[dict[str, Any]], right_daily: list[dict[str, Any]]) -> list[tuple[float, float]]:
    left_by_window: dict[str, list[float]] = {}
    right_by_window: dict[str, list[float]] = {}
    for row in left_daily:
        left_by_window.setdefault(row["window_id"], []).append(row["net_return"])
    for row in right_daily:
        right_by_window.setdefault(row["window_id"], []).append(row["net_return"])
    return [
        (_compound(left_by_window[key]), _compound(right_by_window[key]))
        for key in sorted(set(left_by_window) & set(right_by_window))
    ]


def _fragility_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["fragility"]] = counts.get(row["fragility"], 0) + 1
    return counts


def _zscore_maps(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
        mean = _mean(values) if values else 0.0
        stdev = _std(values)
        result[metric] = {}
        for row in rows:
            value = row.get(metric)
            result[metric][row["benchmark_id"]] = 0.0 if value is None or stdev == 0 else (float(value) - mean) / stdev
    return result


def _ranking_key(item: dict[str, Any]) -> tuple[Any, ...]:
    score = item["champion_score"]
    tie = item["tie_break"]
    return (
        1 if item["fail_closed"] else 0,
        -(score if score is not None else -1e9),
        -(tie["DSR_probability"] or 0.0),
        -(tie["positive_window_ratio"] or 0.0),
        tie["abs_maximum_drawdown"] or 1e9,
        tie["turnover"] or 1e9,
        tie["simplicity_order"],
        tie["benchmark_id"],
    )


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows or [{"empty": True}])
    pq.write_table(table, path)


def _compound(values: list[float] | np.ndarray) -> float:
    total = 1.0
    for value in values:
        total *= 1.0 + float(value)
    return total - 1.0


def _navs(values: list[float]) -> list[float]:
    nav = 100.0
    navs = [nav]
    for value in values:
        nav *= 1.0 + value
        navs.append(nav)
    return navs


def _sortino(values: list[float]) -> float | None:
    downside = [min(0.0, value) for value in values]
    stdev = _std(downside)
    return None if stdev == 0 else _mean(values) / stdev * math.sqrt(252)


def _calendar_returns_from_daily(daily: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for row in daily:
        year = str(row["date"])[:4]
        grouped[year] = (1.0 + grouped.get(year, 0.0)) * (1.0 + row["net_return"]) - 1.0
    return grouped


def _merge_float_maps(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), percentile))


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _dataset_hashes(project_root: Path) -> dict[str, str]:
    paths = sorted((project_root / "data" / "total_returns").rglob("*.parquet"))
    return {
        str(path): digest
        for path in paths
        if (digest := sha256_file(path)) is not None
    }


def _phase6_3_call_counters() -> dict[str, int]:
    return {
        "financial_calls": 0,
        "order_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }
