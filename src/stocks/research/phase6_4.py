from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

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
from stocks.research.phase6_2 import Phase62Layout
from stocks.research.phase6_3 import (
    PAIR_BOOTSTRAP_BLOCK_LENGTH,
    PAIR_BOOTSTRAP_ITERATIONS,
    PAIR_BOOTSTRAP_SEED,
    Phase63Layout,
    alpha_beta_regression,
    block_bootstrap_probability,
    bootstrap_metric_ci,
    deflated_sharpe_probability,
    fragility_classification,
    rejected_strategy_record,
    upside_downside_capture,
)
from stocks.research.phase6_3 import _oos_windows as phase6_3_oos_windows
from stocks.research.phase6_3 import _prepare_common_history_without_import_cycle
from stocks.research.phase6_diagnostics import _detailed_portfolio, _read_json_if_exists


CALCULATION_VERSION = "phase6_4_preregistered_mechanism_research_v1"
PHASE6_4_RANDOM_SEED = PAIR_BOOTSTRAP_SEED
PHASE6_4_MARKER = "PHASE6_4_PREREGISTERED_MECHANISM_RESEARCH_AND_FORWARD_SELECTION_GO"
PHASE6_4_FREEZE_MARKER = "PHASE6_4_PREREGISTERED_MECHANISM_RESEARCH_AND_FORWARD_SELECTION_FROZEN_GO"
PHASE6_4_DECISIONS = (
    "NO_NEW_FINANCIAL_CANDIDATE",
    "PROMISING_MECHANISM_CANDIDATE",
    "FORWARD_RESEARCH_SHADOW_ELIGIBLE",
    "METRIC_OR_DATA_BLOCKED",
)
HYPOTHESIS_IDS = (
    "HYP_A_DIVERSIFIED_DUAL_MOMENTUM",
    "HYP_B_TREND_BREADTH_RISK_ALLOCATION",
    "HYP_C_DIVERSIFIED_TREND_RISK_PARITY",
    "HYP_D_CRISIS_RESILIENT_SLEEVE_ROTATION",
)
ABLATIONS = {
    "HYP_A_DIVERSIFIED_DUAL_MOMENTUM": "without_absolute_trendgate",
    "HYP_B_TREND_BREADTH_RISK_ALLOCATION": "without_breadth_regime_allocation",
    "HYP_C_DIVERSIFIED_TREND_RISK_PARITY": "without_target_volatility_scaling",
    "HYP_D_CRISIS_RESILIENT_SLEEVE_ROTATION": "without_negative_score_filter",
}


@dataclass(frozen=True)
class Phase64Layout:
    project_root: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> Phase64Layout:
        return cls(project_root=project_root)

    @property
    def registry_yaml(self) -> Path:
        return self.project_root / "config" / "research" / "phase6_4_hypotheses.yaml"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output" / "research" / "phase6_4"

    @property
    def preregistration_json(self) -> Path:
        return self.output_dir / "preregistration.json"

    @property
    def hypothesis_results_parquet(self) -> Path:
        return self.output_dir / "hypothesis-results.parquet"

    @property
    def window_results_parquet(self) -> Path:
        return self.output_dir / "window-results.parquet"

    @property
    def decision_log_parquet(self) -> Path:
        return self.output_dir / "decision-log.parquet"

    @property
    def incremental_alpha_parquet(self) -> Path:
        return self.output_dir / "incremental-alpha.parquet"

    @property
    def ablation_results_parquet(self) -> Path:
        return self.output_dir / "ablation-results.parquet"

    @property
    def leave_one_out_parquet(self) -> Path:
        return self.output_dir / "leave-one-out.parquet"

    @property
    def statistical_validation_json(self) -> Path:
        return self.output_dir / "statistical-validation.json"

    @property
    def candidate_ranking_json(self) -> Path:
        return self.output_dir / "candidate-ranking.json"

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


def phase6_4_schema() -> dict[str, Any]:
    return _json_artifact(
        "phase6_4_schema_v1",
        {
            "status": "OFFLINE_SCHEMA_ONLY",
            "technical_marker": PHASE6_4_MARKER,
            "highest_allowed_promotion": "FORWARD_RESEARCH_SHADOW_ELIGIBLE",
            "forbidden_statuses": ["FINANCIAL_FINALIST_GO", "PAPER_STRATEGY_AUTHORITY", "LIVE_STRATEGY_AUTHORITY"],
            "hypothesis_ids": list(HYPOTHESIS_IDS),
            "ablation_count": len(ABLATIONS),
            "selection_count": 8,
            "paired_bootstrap": {
                "iterations": PAIR_BOOTSTRAP_ITERATIONS,
                "block_length": PAIR_BOOTSTRAP_BLOCK_LENGTH,
                "random_seed": PHASE6_4_RANDOM_SEED,
            },
            "decision_statuses": list(PHASE6_4_DECISIONS),
            "financial_calls": _call_counters(),
            "order_calls": 0,
            "market_data_calls": 0,
            "historical_data_calls": 0,
            "account_calls": 0,
        },
        preregistration_hash=None,
        input_hashes={},
    )


def preregister_phase6_4(project_root: Path) -> dict[str, Any]:
    layout = Phase64Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_hypothesis_registry(layout.registry_yaml)
    errors = validate_registry(registry)
    hypothesis_hashes = {item["hypothesis_id"]: hypothesis_hash(item) for item in registry["hypotheses"]}
    stored_hashes = {item["hypothesis_id"]: item.get("parameter_hash") for item in registry["hypotheses"]}
    hash_mismatches = [
        hypothesis_id
        for hypothesis_id, computed in hypothesis_hashes.items()
        if stored_hashes.get(hypothesis_id) != computed
    ]
    registry_hash = stable_hash(registry)
    payload = _json_artifact(
        "phase6_4_preregistration_v1",
        {
            "status": "GO" if not errors and not hash_mismatches else "NO_GO",
            "registry_path": str(layout.registry_yaml),
            "registry_hash": registry_hash,
            "hypothesis_count": len(registry["hypotheses"]),
            "hypothesis_hashes": hypothesis_hashes,
            "hash_mismatches": hash_mismatches,
            "validation_errors": errors,
            "immutable_after_preregistration": True,
            "registry": registry,
        },
        preregistration_hash=registry_hash,
        input_hashes={"registry_yaml": sha256_file(layout.registry_yaml)},
    )
    _write_json(layout.preregistration_json, payload)
    return payload


def run_phase6_4_pipeline(project_root: Path) -> dict[str, Any]:
    layout = Phase64Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = _read_json_if_exists(layout.preregistration_json) or preregister_phase6_4(project_root)
    if preregistration.get("status") != "GO":
        decision = phase6_4_blocked_decision(preregistration.get("registry_hash"), "PREREGISTRATION_INVALID")
        _write_json(layout.decision_json, decision)
        return phase6_4_status(project_root)
    current_registry = load_hypothesis_registry(layout.registry_yaml)
    current_hash = stable_hash(current_registry)
    if current_hash != preregistration["registry_hash"]:
        decision = phase6_4_blocked_decision(preregistration["registry_hash"], "PREREGISTRATION_MUTATED_AFTER_LOCK")
        _write_json(layout.decision_json, decision)
        return phase6_4_status(project_root)

    dataset = load_phase6_dataset(project_root)
    prepared = _prepare_common_history_without_import_cycle(dataset)
    windows = phase6_4_windows(project_root, prepared)
    benchmarks = evaluate_phase6_4_benchmarks(prepared, windows, preregistration["registry_hash"])
    diversified_champion = choose_diversified_benchmark_champion(benchmarks)
    reports = [
        evaluate_hypothesis(item, prepared, windows, preregistration["registry_hash"])
        for item in current_registry["hypotheses"]
    ]
    stats = statistical_validation(reports, preregistration["registry_hash"])
    for report in reports:
        stat = next(row for row in stats["rows"] if row["hypothesis_id"] == report["hypothesis_id"])
        report["metrics"]["DSR_probability"] = stat["DSR_probability"]
        report["metrics"]["PBO"] = stat["PBO"]
        report["metrics"]["bootstrap_probability_return_gt_0"] = stat["bootstrap_probability_return_gt_0"]
    alpha_rows = incremental_alpha_rows(reports, diversified_champion, benchmarks["BUY_AND_HOLD_GLD"], preregistration["registry_hash"])
    ablation_rows = ablation_results(current_registry["hypotheses"], reports, prepared, windows, preregistration["registry_hash"])
    loo_rows = leave_one_out_rows(current_registry["hypotheses"], reports, prepared, windows, preregistration["registry_hash"])
    ranking = candidate_ranking(reports, alpha_rows, ablation_rows, preregistration["registry_hash"])
    decision = phase6_4_decision(ranking, reports, alpha_rows, ablation_rows, loo_rows, preregistration["registry_hash"])

    _write_parquet(layout.hypothesis_results_parquet, [_hypothesis_result_row(report, preregistration["registry_hash"]) for report in reports])
    _write_parquet(layout.window_results_parquet, [row for report in reports for row in report["window_rows"]])
    _write_parquet(layout.decision_log_parquet, [row for report in reports for row in report["decision_rows"]])
    _write_parquet(layout.incremental_alpha_parquet, alpha_rows)
    _write_parquet(layout.ablation_results_parquet, ablation_rows)
    _write_parquet(layout.leave_one_out_parquet, loo_rows)
    _write_json(layout.statistical_validation_json, stats)
    _write_json(layout.candidate_ranking_json, ranking)
    _write_json(layout.decision_json, decision)
    if decision["financial_decision"] == "FORWARD_RESEARCH_SHADOW_ELIGIBLE":
        best = next(report for report in reports if report["hypothesis_id"] == decision["selected_hypothesis_id"])
        _write_json(layout.forward_shadow_spec_json, forward_shadow_spec(project_root, best, preregistration["registry_hash"]))
    elif layout.forward_shadow_spec_json.exists():
        layout.forward_shadow_spec_json.unlink()
    manifest = phase6_4_manifest(project_root, preregistration, reports, benchmarks, diversified_champion, decision)
    _write_json(layout.manifest_json, manifest)
    return phase6_4_status(project_root, manifest=manifest, decision=decision)


def phase6_4_status(
    project_root: Path,
    *,
    manifest: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layout = Phase64Layout.from_project_root(project_root)
    manifest = manifest or _read_json_if_exists(layout.manifest_json)
    decision = decision or _read_json_if_exists(layout.decision_json)
    artifacts = required_artifacts(layout)
    checks = {
        "preregistration_available": layout.preregistration_json.exists(),
        "required_artifacts_exist": all(path.exists() for path in artifacts.values()),
        "manifest_available": manifest is not None,
        "technical_marker_go": manifest is not None and manifest.get("technical_marker") == PHASE6_4_MARKER,
        "decision_available": decision is not None and decision.get("financial_decision") in PHASE6_4_DECISIONS,
        "four_hypotheses": manifest is not None and manifest.get("summary", {}).get("hypothesis_count") == 4,
        "four_ablations": manifest is not None and manifest.get("summary", {}).get("ablation_count") == 4,
        "old_strategy_rejected": manifest is not None
        and manifest.get("strategy_rejection", {}).get("strategy_status") == "REJECTED_NO_INCREMENTAL_FINANCIAL_EDGE",
        "gld_not_diversified_champion": manifest is not None
        and manifest.get("diversified_benchmark_champion") != "BUY_AND_HOLD_GLD",
        "no_authority": decision is not None
        and decision.get("FINANCIAL_FINALIST_GO") is False
        and decision.get("PAPER_STRATEGY_AUTHORITY") == "blocked"
        and decision.get("LIVE_STRATEGY_AUTHORITY") == "blocked",
        "counters_zero": manifest is not None and manifest.get("call_counters") == _call_counters(),
    }
    status = PHASE6_4_MARKER if all(checks.values()) else "NO_GO"
    return _json_artifact(
        "phase6_4_status_v1",
        {
            "status": status,
            "financial_decision": decision.get("financial_decision") if decision else "METRIC_OR_DATA_BLOCKED",
            "checks": checks,
            "summary": (manifest or {}).get("summary", {}),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "financial_calls": _call_counters(),
            "order_calls": 0,
            "market_data_calls": 0,
            "historical_data_calls": 0,
            "account_calls": 0,
        },
        preregistration_hash=(manifest or {}).get("preregistration_hash"),
        input_hashes=(manifest or {}).get("input_hashes", {}),
    )


def phase6_4_freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase64Layout.from_project_root(project_root)
    status = phase6_4_status(project_root)
    artifact_paths = list(required_artifacts(layout).values())
    if layout.forward_shadow_spec_json.exists():
        artifact_paths.append(layout.forward_shadow_spec_json)
    source_paths = [
        "src/stocks/research/phase6_4.py",
        "src/stocks/research/phase6_3.py",
        "src/stocks/research/phase6_2.py",
        "src/stocks/research/phase6.py",
        "config/research/phase6_4_hypotheses.yaml",
        "main.py",
        "tests/test_phase6_4.py",
    ]
    freeze = _json_artifact(
        "phase6_4_freeze_status_v1",
        {
            "freeze_status": PHASE6_4_FREEZE_MARKER if status["status"] == PHASE6_4_MARKER else "NO_GO",
            "phase6_4_status": status["status"],
            "financial_decision": status["financial_decision"],
            "source_hashes": {path: sha256_file(project_root / path) for path in source_paths if (project_root / path).exists()},
            "artifact_hashes": {str(path): sha256_file(path) for path in artifact_paths if path.exists()},
            "financial_calls": _call_counters(),
            "order_calls": 0,
            "market_data_calls": 0,
            "historical_data_calls": 0,
            "account_calls": 0,
        },
        preregistration_hash=status.get("preregistration_hash"),
        input_hashes=status.get("input_hashes", {}),
    )
    _write_json(layout.freeze_json, freeze)
    return freeze


def load_hypothesis_registry(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    hypotheses = registry.get("hypotheses", [])
    if [item.get("hypothesis_id") for item in hypotheses] != list(HYPOTHESIS_IDS):
        errors.append("registry must contain exactly the four preregistered hypotheses in canonical order")
    required = {
        "hypothesis_id",
        "version",
        "economic_rationale",
        "expected_mechanism",
        "expected_holding_period",
        "expected_market_regimes",
        "expected_failure_regimes",
        "eligible_universe_rule",
        "signal_definition",
        "rebalance_rule",
        "weighting_rule",
        "cash_rule",
        "risk_constraints",
        "transaction_costs",
        "maximum_turnover",
        "comparison_benchmarks",
        "null_hypothesis",
        "success_criteria",
        "rejection_criteria",
        "parameter_hash",
    }
    for item in hypotheses:
        missing = sorted(required - set(item))
        if missing:
            errors.append(f"{item.get('hypothesis_id', 'UNKNOWN')}: missing {missing}")
    return errors


def hypothesis_hash(hypothesis: dict[str, Any]) -> str:
    payload = {key: value for key, value in hypothesis.items() if key != "parameter_hash"}
    return stable_hash(payload)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest().upper()


def phase6_4_windows(project_root: Path, prepared: dict[str, Any]) -> list[dict[str, Any]]:
    base = phase6_3_oos_windows(project_root)
    windows = [dict(window, evaluation_method=window["kind"]) for window in base]
    dates = prepared["dates"]
    for start_year in range(2021, 2026):
        start = "2021-01-01"
        end = f"{start_year}-12-31"
        if end <= dates[-1]:
            windows.append({"window_id": f"anchored_{start_year}", "kind": "anchored", "evaluation_method": "anchored_expanding", "test_start": start, "test_end": end})
    for start_year in range(2018, 2024):
        start = f"{start_year}-01-01"
        end = f"{start_year + 2}-12-31"
        if end <= dates[-1]:
            windows.append({"window_id": f"rolling3y_{start_year}", "kind": "rolling3y", "evaluation_method": "rolling_3_year", "test_start": start, "test_end": end})
    for shift, label in ((5, "start_plus_5"), (10, "start_plus_10")):
        for window in base[:5]:
            start = _shift_to_existing_date(dates, window["test_start"], shift)
            windows.append({**window, "window_id": f"{label}_{window['window_id']}", "evaluation_method": "start_date_perturbation", "test_start": start})
    for offset in (1, 2):
        for window in base[:5]:
            windows.append({**window, "window_id": f"rebalance_offset_{offset}_{window['window_id']}", "evaluation_method": "rebalance_date_perturbation", "rebalance_offset": offset})
    return sorted(windows, key=lambda item: (item["test_start"], item["window_id"]))


def evaluate_phase6_4_benchmarks(prepared: dict[str, Any], windows: list[dict[str, Any]], preregistration_hash: str) -> dict[str, dict[str, Any]]:
    gl_key = _symbol_key(prepared, "GLD")
    cash_key = _cash_key(prepared)
    candidates = [
        ("EQUAL_WEIGHT", lambda subset, _offset=0: _equal_weight_targets(subset, "monthly")),
        ("INVERSE_VOLATILITY", lambda subset, _offset=0: _inverse_vol_targets(subset, lookback=63, rebalance="monthly")),
        ("TREND_200D", lambda subset, _offset=0: _trend_targets(subset, trend_lookback=200, rebalance="monthly")),
        ("MOMENTUM_ROTATION", lambda subset, _offset=0: _momentum_targets(subset, momentum_lookback=252, trend_lookback=200, top_n=4, rebalance="monthly")),
        ("BUY_AND_HOLD_GLD", lambda subset, _offset=0: [{gl_key: 1.0} for _day in subset["dates"]]),
        ("CASH_BIL", lambda subset, _offset=0: [{cash_key: 1.0} for _day in subset["dates"]]),
    ]
    return {
        name: evaluate_named_strategy(
            strategy_id=name,
            family="BENCHMARK",
            prepared=prepared,
            windows=windows,
            target_builder=builder,
            preregistration_hash=preregistration_hash,
        )
        for name, builder in candidates
    }


def choose_diversified_benchmark_champion(benchmarks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        report
        for name, report in benchmarks.items()
        if name != "BUY_AND_HOLD_GLD"
        and report["metrics"]["single_asset_contribution_max"] < 0.40
        and report["metrics"]["single_region_contribution_max"] < 0.60
        and report["metrics"]["single_sleeve_contribution_max"] < 0.70
    ]
    if not eligible:
        return benchmarks["EQUAL_WEIGHT"]
    return sorted(
        eligible,
        key=lambda item: (
            -(item["metrics"]["Sharpe"] or -999),
            -(item["metrics"]["aggregate_oos_CAGR"] or -999),
            item["strategy_id"],
        ),
    )[0]


def evaluate_hypothesis(hypothesis: dict[str, Any], prepared: dict[str, Any], windows: list[dict[str, Any]], preregistration_hash: str) -> dict[str, Any]:
    hypothesis_id = hypothesis["hypothesis_id"]
    return evaluate_named_strategy(
        strategy_id=hypothesis_id,
        family="HYPOTHESIS",
        prepared=prepared,
        windows=windows,
        target_builder=lambda subset, offset=0, hid=hypothesis_id: hypothesis_targets(hid, subset, rebalance_offset=offset),
        preregistration_hash=preregistration_hash,
        registry_hypothesis=hypothesis,
    )


def evaluate_named_strategy(
    *,
    strategy_id: str,
    family: str,
    prepared: dict[str, Any],
    windows: list[dict[str, Any]],
    target_builder: Any,
    preregistration_hash: str,
    registry_hypothesis: dict[str, Any] | None = None,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    daily: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    contributions_asset: dict[str, float] = {}
    contributions_year: dict[str, float] = {}
    nav = 100.0
    for window in windows:
        subset = _slice_prepared(prepared, window["test_start"], window["test_end"])
        offset = int(window.get("rebalance_offset", 0))
        targets = target_builder(subset, offset)
        detailed = _detailed_portfolio(subset, targets, cost_bps=cost_bps, parameters={"strategy_id": strategy_id}, name=strategy_id)
        returns = [row["net_return"] for row in detailed["daily"]]
        for row in detailed["daily"]:
            nav *= 1.0 + row["net_return"]
            daily.append({**row, "obs_id": f"{window['window_id']}:{row['date']}", "window_id": window["window_id"], "strategy_id": strategy_id, "nav": nav})
        for episode in detailed["episodes"]:
            episodes.append({**episode, "window_id": window["window_id"]})
        for index, weights in enumerate(targets):
            if index == 0 or not weights:
                continue
            day = subset["dates"][index]
            next_day = subset["dates"][index + 1] if index + 1 < len(subset["dates"]) else None
            decision_rows.append(
                _parquet_row(
                    {
                        "strategy_id": strategy_id,
                        "signal_timestamp": f"{day}T21:00:00Z",
                        "decision_timestamp": f"{day}T21:05:00Z",
                        "first_executable_timestamp": f"{next_day}T14:30:00Z" if next_day else None,
                        "dataset_version": "phase5_total_return_fx_v1",
                        "eligible_universe": json.dumps(sorted(subset["metadata"]), sort_keys=True),
                        "blocked_instruments": "[]",
                        "block_reasons": "{}",
                        "target_weights": json.dumps(weights, sort_keys=True),
                        "causality_status": "GO" if next_day is None or next_day > day else "NO_GO",
                    },
                    preregistration_hash,
                    {},
                )
            )
        _merge(contributions_asset, detailed["contribution_by_asset"])
        _merge(contributions_year, detailed["contribution_by_year"])
        window_rows.append(
            _parquet_row(
                {
                    "hypothesis_id": strategy_id,
                    "strategy_id": strategy_id,
                    "evaluation_method": window["evaluation_method"],
                    "window_id": window["window_id"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "window_return": _compound(returns),
                    "window_pf": _profit_factor(returns),
                    "closed_episodes": len(detailed["episodes"]),
                    "turnover": sum(row["turnover"] for row in detailed["daily"]),
                    "transaction_costs": sum(row["cost"] for row in detailed["daily"]),
                },
                preregistration_hash,
                {},
            )
        )
    metrics = aggregate_metrics(strategy_id, prepared, daily, episodes, window_rows, contributions_asset, contributions_year, preregistration_hash)
    stress = {cost: stress_pf(strategy_id, prepared, windows, target_builder, cost) for cost in (20.0, 30.0, 50.0)}
    metrics["20bps_stress_pf"] = stress[20.0]
    metrics["30bps_stress_pf"] = stress[30.0]
    metrics["50bps_stress_pf"] = stress[50.0]
    return {
        "strategy_id": strategy_id,
        "hypothesis_id": strategy_id,
        "family": family,
        "registry_hypothesis": registry_hypothesis,
        "metrics": metrics,
        "daily": daily,
        "episodes": episodes,
        "window_rows": window_rows,
        "decision_rows": decision_rows,
        "contribution_by_asset": contributions_asset,
        "contribution_by_year": contributions_year,
    }


def aggregate_metrics(
    strategy_id: str,
    prepared: dict[str, Any],
    daily: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    contribution_by_asset: dict[str, float],
    contribution_by_year: dict[str, float],
    preregistration_hash: str,
) -> dict[str, Any]:
    returns = [row["net_return"] for row in daily]
    turnovers = [row["turnover"] for row in daily]
    costs = [row["cost"] for row in daily]
    cash = [row["cash_weight"] for row in daily]
    navs = _navs(returns)
    total_return = navs[-1] / navs[0] - 1.0 if len(navs) > 1 else 0.0
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 and years > 0 else None
    ep = episode_pf([row["pnl"] for row in episodes])
    dominance = dominance_metrics(prepared, contribution_by_asset, contribution_by_year)
    window_returns = [float(row["window_return"]) for row in window_rows]
    window_pfs = [float(row["window_pf"]) for row in window_rows if row.get("window_pf") is not None]
    gross = _compound([row["gross_return"] for row in daily])
    return {
        "hypothesis_id": strategy_id,
        "testwindow_count": len(window_rows),
        "evaluable_testwindow_count": sum(1 for row in window_rows if row["closed_episodes"] > 0 or row["window_return"] != 0),
        "positive_testwindow_count": sum(1 for value in window_returns if value > 0),
        "positive_testwindow_ratio": sum(1 for value in window_returns if value > 0) / len(window_returns) if window_returns else 0.0,
        "aggregate_oos_return": total_return,
        "aggregate_oos_CAGR": cagr,
        "annualized_volatility": _std(returns) * math.sqrt(252),
        "Sharpe": None if _std(returns) == 0 else _mean(returns) / _std(returns) * math.sqrt(252),
        "Sortino": sortino(returns),
        "maximum_drawdown": _max_drawdown(navs),
        "Calmar": None if not cagr or _max_drawdown(navs) == 0 else cagr / abs(_max_drawdown(navs)),
        "period_profit_factor": _profit_factor(returns),
        "episode_profit_factor": ep["episode_pf"],
        "pf_status": ep["pf_status"],
        "closed_episodes": len(episodes),
        "turnover": sum(turnovers),
        "transaction_costs": sum(costs),
        "average_cash_weight": _mean(cash) if cash else 0.0,
        "maximum_cash_weight": max(cash) if cash else 0.0,
        "worst_window_return": min(window_returns) if window_returns else None,
        "worst_window_pf": min(window_pfs) if window_pfs else None,
        "best_window_return": max(window_returns) if window_returns else None,
        "break_even_cost_bps": None if sum(turnovers) == 0 else gross / sum(turnovers) * 10_000,
        **dominance,
        "DSR_probability": None,
        "PBO": None,
        "bootstrap_probability_return_gt_0": None,
        "bootstrap_probability_beat_diversified_champion": None,
        "bootstrap_probability_beat_GLD": None,
        "preregistration_hash": preregistration_hash,
    }


def hypothesis_targets(hypothesis_id: str, prepared: dict[str, Any], *, rebalance_offset: int = 0, ablation: str | None = None) -> list[dict[str, float]]:
    if hypothesis_id == "HYP_A_DIVERSIFIED_DUAL_MOMENTUM":
        return dual_momentum_targets(prepared, rebalance_offset=rebalance_offset, use_trend=ablation != "without_absolute_trendgate")
    if hypothesis_id == "HYP_B_TREND_BREADTH_RISK_ALLOCATION":
        return breadth_targets(prepared, rebalance_offset=rebalance_offset, use_regimes=ablation != "without_breadth_regime_allocation")
    if hypothesis_id == "HYP_C_DIVERSIFIED_TREND_RISK_PARITY":
        return trend_risk_parity_targets(prepared, rebalance_offset=rebalance_offset, use_target_vol=ablation != "without_target_volatility_scaling")
    if hypothesis_id == "HYP_D_CRISIS_RESILIENT_SLEEVE_ROTATION":
        return sleeve_rotation_targets(prepared, rebalance_offset=rebalance_offset, filter_negative=ablation != "without_negative_score_filter")
    raise ValueError(f"unknown hypothesis_id: {hypothesis_id}")


def dual_momentum_targets(prepared: dict[str, Any], *, rebalance_offset: int = 0, use_trend: bool = True) -> list[dict[str, float]]:
    targets: list[dict[str, float]] = []
    previous: dict[str, float] = {}
    risk_keys = [key for key, meta in prepared["metadata"].items() if meta["sleeve"] in {"equity", "commodity"}]
    cash = _cash_key(prepared)
    for index, _day in enumerate(prepared["dates"]):
        if index < 252 or not is_monthly_rebalance(prepared["dates"], index, rebalance_offset):
            targets.append(previous)
            continue
        scored = []
        for key in risk_keys:
            prices = prepared["prices"][key]
            trend_ok = prices[index] > _mean(prices[index - 200 : index]) if use_trend else True
            if prices[index - 252] > 0 and trend_ok:
                score = prices[index - 21] / prices[index - 252] - 1
                if score > 0:
                    scored.append((score, key))
        selected = [key for _score, key in sorted(scored, reverse=True)[:3]]
        previous = cap_weights({key: 1 / len(selected) for key in selected}, prepared, instrument_cap=0.35, region_cap=0.50, sleeve_cap=0.60, cash_key=cash) if selected else {cash: 1.0}
        targets.append(previous)
    return targets


def breadth_targets(prepared: dict[str, Any], *, rebalance_offset: int = 0, use_regimes: bool = True) -> list[dict[str, float]]:
    targets: list[dict[str, float]] = []
    previous: dict[str, float] = {}
    risk = [key for key, meta in prepared["metadata"].items() if meta["sleeve"] in {"equity", "commodity"} and meta["symbol"] != "GLD"]
    bonds = [key for key, meta in prepared["metadata"].items() if meta["sleeve"] == "defensive"]
    gold_key = _optional_symbol_key(prepared, "GLD")
    gold = [gold_key] if gold_key else []
    cash = _cash_key(prepared)
    for index, _day in enumerate(prepared["dates"]):
        if index < 200 or not is_monthly_rebalance(prepared["dates"], index, rebalance_offset):
            targets.append(previous)
            continue
        breadth = trend_breadth(prepared, risk, index)
        if not use_regimes:
            sleeve_budget = {"risk": 0.50, "bonds": 0.25, "gold": 0.15, "cash": 0.10}
        elif breadth >= 0.70:
            sleeve_budget = {"risk": 0.80, "bonds": 0.10, "gold": 0.00, "cash": 0.10}
        elif breadth >= 0.40:
            sleeve_budget = {"risk": 0.50, "bonds": 0.25, "gold": 0.15, "cash": 0.10}
        else:
            sleeve_budget = {"risk": 0.20, "bonds": 0.30, "gold": 0.30, "cash": 0.20}
        weights: dict[str, float] = {}
        weights.update(scale_weights(inverse_vol_weights(prepared, risk, index, 63), sleeve_budget["risk"]))
        weights.update(scale_weights(inverse_vol_weights(prepared, bonds, index, 63), sleeve_budget["bonds"]))
        weights.update(scale_weights(inverse_vol_weights(prepared, gold, index, 63), sleeve_budget["gold"]))
        weights[cash] = weights.get(cash, 0.0) + sleeve_budget["cash"] + max(0.0, 1.0 - sum(weights.values()) - sleeve_budget["cash"])
        previous = normalize(weights)
        targets.append(previous)
    return targets


def trend_risk_parity_targets(prepared: dict[str, Any], *, rebalance_offset: int = 0, use_target_vol: bool = True) -> list[dict[str, float]]:
    targets: list[dict[str, float]] = []
    previous: dict[str, float] = {}
    cash = _cash_key(prepared)
    keys = [key for key in prepared["metadata"] if key != cash]
    for index, _day in enumerate(prepared["dates"]):
        if index < 200 or not is_monthly_rebalance(prepared["dates"], index, rebalance_offset):
            targets.append(previous)
            continue
        eligible = [key for key in keys if prepared["prices"][key][index] > _mean(prepared["prices"][key][index - 200 : index])]
        weights = inverse_vol_weights(prepared, eligible, index, 63)
        weights = cap_weights(weights, prepared, instrument_cap=0.30, region_cap=0.50, sleeve_cap=0.55, cash_key=cash)
        if use_target_vol and weights:
            recent = [sum(weights.get(key, 0.0) * prepared["returns"][key][j] for key in weights if key in prepared["returns"]) for j in range(max(1, index - 63), index)]
            realized = _std(recent) * math.sqrt(252)
            scale = min(1.0, 0.10 / realized) if realized > 0 else 1.0
            weights = {key: value * scale for key, value in weights.items()}
        weights[cash] = max(0.0, 1.0 - sum(weights.values()))
        previous = normalize(weights)
        targets.append(previous)
    return targets


def sleeve_rotation_targets(prepared: dict[str, Any], *, rebalance_offset: int = 0, filter_negative: bool = True) -> list[dict[str, float]]:
    targets: list[dict[str, float]] = []
    previous: dict[str, float] = {}
    cash = _cash_key(prepared)
    sleeves = ["equity", "defensive", "commodity"]
    for index, _day in enumerate(prepared["dates"]):
        if index < 252 or not is_monthly_rebalance(prepared["dates"], index, rebalance_offset):
            targets.append(previous)
            continue
        raw_scores = {sleeve: sleeve_raw_score(prepared, sleeve, index) for sleeve in sleeves}
        values = list(raw_scores.values())
        mean = _mean(values)
        stdev = _std(values) or 1.0
        scores = {sleeve: (value - mean) / stdev for sleeve, value in raw_scores.items()}
        selected = [sleeve for sleeve, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score > 0 or not filter_negative][:2]
        if not selected:
            previous = {cash: 1.0}
            targets.append(previous)
            continue
        weights: dict[str, float] = {cash: 0.10}
        sleeve_budget = min(0.55, 0.90 / len(selected))
        for sleeve in selected:
            keys = [key for key, meta in prepared["metadata"].items() if meta["sleeve"] == sleeve and key != cash]
            weights.update(scale_weights(inverse_vol_weights(prepared, keys, index, 63), sleeve_budget))
        weights[cash] = weights.get(cash, 0.0) + max(0.0, 1.0 - sum(weights.values()))
        previous = normalize(weights)
        targets.append(previous)
    return targets


def ablation_results(hypotheses: list[dict[str, Any]], full_reports: list[dict[str, Any]], prepared: dict[str, Any], windows: list[dict[str, Any]], preregistration_hash: str) -> list[dict[str, Any]]:
    rows = []
    full_by_id = {report["hypothesis_id"]: report for report in full_reports}
    for hypothesis in hypotheses:
        hypothesis_id = hypothesis["hypothesis_id"]
        ablation = ABLATIONS[hypothesis_id]
        report = evaluate_named_strategy(
            strategy_id=f"{hypothesis_id}_{ablation}",
            family="ABLATION",
            prepared=prepared,
            windows=windows,
            target_builder=lambda subset, offset=0, hid=hypothesis_id, abl=ablation: hypothesis_targets(hid, subset, rebalance_offset=offset, ablation=abl),
            preregistration_hash=preregistration_hash,
        )
        full = full_by_id[hypothesis_id]["metrics"]
        ablated = report["metrics"]
        row = _parquet_row(
            {
                "hypothesis_id": hypothesis_id,
                "ablation_id": ablation,
                "full_model_metric": full["aggregate_oos_CAGR"],
                "ablation_metric": ablated["aggregate_oos_CAGR"],
                "incremental_CAGR": _delta(full["aggregate_oos_CAGR"], ablated["aggregate_oos_CAGR"]),
                "incremental_Sharpe": _delta(full["Sharpe"], ablated["Sharpe"]),
                "incremental_drawdown_improvement": abs(ablated["maximum_drawdown"] or 0) - abs(full["maximum_drawdown"] or 0),
                "incremental_period_pf": _delta(full["period_profit_factor"], ablated["period_profit_factor"]),
                "incremental_turnover": _delta(full["turnover"], ablated["turnover"]),
                "component_status": component_status(full, ablated),
            },
            preregistration_hash,
            {},
        )
        rows.append(row)
    return rows


def leave_one_out_rows(hypotheses: list[dict[str, Any]], reports: list[dict[str, Any]], prepared: dict[str, Any], windows: list[dict[str, Any]], preregistration_hash: str) -> list[dict[str, Any]]:
    rows = []
    metadata = prepared["metadata"]
    removals = (
        [("instrument", key) for key in sorted(metadata)]
        + [("region", value) for value in sorted({meta["region"] for meta in metadata.values()})]
        + [("sleeve", value) for value in sorted({meta["sleeve"] for meta in metadata.values()})]
    )
    full_by_id = {report["hypothesis_id"]: report for report in reports}
    for hypothesis in hypotheses:
        hypothesis_id = hypothesis["hypothesis_id"]
        full = full_by_id[hypothesis_id]["metrics"]
        for entity_type, entity in removals:
            reduced = remove_entity(prepared, entity_type, entity)
            if len(reduced["returns"]) < 2:
                continue
            report = evaluate_named_strategy(
                strategy_id=hypothesis_id,
                family="HYPOTHESIS_LOO",
                prepared=reduced,
                windows=windows,
                target_builder=lambda subset, offset=0, hid=hypothesis_id: hypothesis_targets(hid, subset, rebalance_offset=offset),
                preregistration_hash=preregistration_hash,
            )
            metrics = report["metrics"]
            rows.append(
                _parquet_row(
                    {
                        "hypothesis_id": hypothesis_id,
                        "removed_entity": str(entity),
                        "entity_type": entity_type,
                        "CAGR_difference": _delta(full["aggregate_oos_CAGR"], metrics["aggregate_oos_CAGR"]),
                        "Sharpe_difference": _delta(full["Sharpe"], metrics["Sharpe"]),
                        "PF_difference": _delta(full["period_profit_factor"], metrics["period_profit_factor"]),
                        "drawdown_difference": _delta(full["maximum_drawdown"], metrics["maximum_drawdown"]),
                        "candidate_status_after_removal": candidate_basic_status(metrics),
                        "fragility_status": fragility_classification(full["aggregate_oos_CAGR"], metrics["aggregate_oos_CAGR"]),
                    },
                    preregistration_hash,
                    {},
                )
            )
    return rows


def incremental_alpha_rows(reports: list[dict[str, Any]], diversified_champion: dict[str, Any], gld: dict[str, Any], preregistration_hash: str) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        for benchmark_name, benchmark in (
            ("DIVERSIFIED_BENCHMARK_CHAMPION", diversified_champion),
            ("BUY_AND_HOLD_GLD", gld),
        ):
            rows.append(incremental_alpha_row(report, benchmark, benchmark_name, preregistration_hash))
    by_hyp: dict[str, dict[str, Any]] = {
        str(row["hypothesis_id"]): {} for row in rows
    }
    for row in rows:
        by_hyp[row["hypothesis_id"]][row["benchmark_id"]] = row["paired_bootstrap_probability_strategy_gt_benchmark"]
    for report in reports:
        metrics = report["metrics"]
        metrics["bootstrap_probability_beat_diversified_champion"] = by_hyp[report["hypothesis_id"]].get("DIVERSIFIED_BENCHMARK_CHAMPION")
        metrics["bootstrap_probability_beat_GLD"] = by_hyp[report["hypothesis_id"]].get("BUY_AND_HOLD_GLD")
    return rows


def incremental_alpha_row(report: dict[str, Any], benchmark: dict[str, Any], benchmark_id: str, preregistration_hash: str) -> dict[str, Any]:
    left = {row["obs_id"]: row for row in report["daily"]}
    right = {row["obs_id"]: row for row in benchmark["daily"]}
    obs_ids = sorted(set(left) & set(right))
    strategy_returns = [left[obs]["net_return"] for obs in obs_ids]
    benchmark_returns = [right[obs]["net_return"] for obs in obs_ids]
    active = [left_value - right_value for left_value, right_value in zip(strategy_returns, benchmark_returns, strict=True)]
    tracking_error = _std(active) * math.sqrt(252)
    regression = alpha_beta_regression(strategy_returns, benchmark_returns)
    capture = upside_downside_capture(strategy_returns, benchmark_returns)
    window_pairs = paired_window_returns(report["daily"], benchmark["daily"])
    window_active = [left_value - right_value for left_value, right_value in window_pairs]
    return _parquet_row(
        {
            "hypothesis_id": report["hypothesis_id"],
            "benchmark_id": benchmark_id,
            "overlapping_observations": len(obs_ids),
            "annualized_active_return": _mean(active) * 252 if active else None,
            "tracking_error": tracking_error,
            "information_ratio": None if tracking_error == 0 else _mean(active) / _std(active) * math.sqrt(252),
            "alpha_annualized": None if regression["alpha_daily"] is None else regression["alpha_daily"] * 252,
            "alpha_t_stat": regression["alpha_t_stat"],
            "beta": regression["beta"],
            "upside_capture": capture["upside_capture"],
            "downside_capture": capture["downside_capture"],
            "active_maximum_drawdown": _max_drawdown(_navs(active)) if active else None,
            "paired_daily_win_rate": sum(1 for value in active if value > 0) / len(active) if active else None,
            "paired_window_win_rate": sum(1 for value in window_active if value > 0) / len(window_active) if window_active else None,
            "paired_bootstrap_probability_strategy_gt_benchmark": block_bootstrap_probability(active),
        },
        preregistration_hash,
        {},
    )


def statistical_validation(reports: list[dict[str, Any]], preregistration_hash: str) -> dict[str, Any]:
    sharpes = [report["metrics"]["Sharpe"] for report in reports if report["metrics"]["Sharpe"] is not None]
    median_sharpe = _median(sharpes) if sharpes else None
    threshold = 0.0 if median_sharpe is None else median_sharpe
    rows = []
    for index, report in enumerate(reports):
        returns = [row["net_return"] for row in report["daily"]]
        bootstrap = bootstrap_metric_ci(returns, iterations=1000, seed=9000 + index)
        pbo = pbo_for_report(report, reports)
        raw_sharpe = report["metrics"]["Sharpe"]
        dsr = (
            None
            if raw_sharpe is None
            else deflated_sharpe_probability(raw_sharpe, threshold, 8)
        )
        rows.append(
            {
                "hypothesis_id": report["hypothesis_id"],
                "raw_Sharpe": report["metrics"]["Sharpe"],
                "DSR_probability": dsr,
                "PBO": pbo,
                "bootstrap_CAGR_ci": bootstrap["CAGR_ci"],
                "bootstrap_Sharpe_ci": bootstrap["Sharpe_ci"],
                "bootstrap_probability_return_gt_0": bootstrap["probability_total_return_gt_0"],
                "false_discovery_aware_candidate_status": (
                    "WEAK_AFTER_SELECTION_CORRECTION"
                    if dsr is None or dsr < 0.95
                    else "SELECTION_CORRECTION_GO"
                ),
            }
        )
    return _json_artifact(
        "phase6_4_statistical_validation_v1",
        {
            "status": "GO",
            "selection_count": 8,
            "prior_strategy_configurations": 108,
            "prior_benchmark_variants": phase6_3_variant_count(),
            "rows": rows,
            "financial_calls": _call_counters(),
        },
        preregistration_hash=preregistration_hash,
        input_hashes={},
    )


def candidate_ranking(reports: list[dict[str, Any]], alpha_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]], preregistration_hash: str) -> dict[str, Any]:
    alpha_by_hyp = {(row["hypothesis_id"], row["benchmark_id"]): row for row in alpha_rows}
    ablation_by_hyp = {row["hypothesis_id"]: row for row in ablation_rows}
    ranked = []
    for report in reports:
        metrics = report["metrics"]
        alpha_div = alpha_by_hyp[(report["hypothesis_id"], "DIVERSIFIED_BENCHMARK_CHAMPION")]
        score = (
            0.25 * _finite(metrics["aggregate_oos_CAGR"])
            + 0.20 * _finite(metrics["Sharpe"])
            + 0.15 * _finite(metrics["period_profit_factor"])
            + 0.15 * _finite(metrics["positive_testwindow_ratio"])
            + 0.10 * _finite(metrics["20bps_stress_pf"])
            + 0.10 * _finite(alpha_div["paired_bootstrap_probability_strategy_gt_benchmark"])
            - 0.05 * abs(_finite(metrics["maximum_drawdown"]))
        )
        ranked.append(
            {
                "hypothesis_id": report["hypothesis_id"],
                "score": score,
                "aggregate_oos_CAGR": metrics["aggregate_oos_CAGR"],
                "Sharpe": metrics["Sharpe"],
                "period_profit_factor": metrics["period_profit_factor"],
                "positive_testwindow_ratio": metrics["positive_testwindow_ratio"],
                "component_status": ablation_by_hyp[report["hypothesis_id"]]["component_status"],
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["hypothesis_id"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return _json_artifact(
        "phase6_4_candidate_ranking_v1",
        {"status": "GO", "ranking": ranked, "best_hypothesis_id": ranked[0]["hypothesis_id"] if ranked else None},
        preregistration_hash=preregistration_hash,
        input_hashes={},
    )


def phase6_4_decision(
    ranking: dict[str, Any],
    reports: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    loo_rows: list[dict[str, Any]],
    preregistration_hash: str,
) -> dict[str, Any]:
    if not ranking["ranking"]:
        return phase6_4_blocked_decision(preregistration_hash, "NO_RANKING")
    best_id = ranking["ranking"][0]["hypothesis_id"]
    report = next(item for item in reports if item["hypothesis_id"] == best_id)
    metrics = report["metrics"]
    ablation = next(row for row in ablation_rows if row["hypothesis_id"] == best_id)
    loo = [row for row in loo_rows if row["hypothesis_id"] == best_id]
    alpha_div = next(row for row in alpha_rows if row["hypothesis_id"] == best_id and row["benchmark_id"] == "DIVERSIFIED_BENCHMARK_CHAMPION")
    promising = {
        "positive_testwindow_ratio_ge_60": metrics["positive_testwindow_ratio"] >= 0.60,
        "aggregate_oos_period_pf_gt_1_05": (metrics["period_profit_factor"] or 0) > 1.05,
        "20bps_stress_pf_gt_1": (metrics["20bps_stress_pf"] or 0) > 1.0,
        "aggregate_oos_CAGR_gt_0": (metrics["aggregate_oos_CAGR"] or 0) > 0,
        "Sharpe_gt_0": (metrics["Sharpe"] or 0) > 0,
        "no_blocking_concentration": not concentration_blocked(metrics),
        "component_not_harmful": ablation["component_status"] != "HARMFUL",
        "bootstrap_probability_return_gt_0_ge_80": (metrics["bootstrap_probability_return_gt_0"] or 0) >= 0.80,
    }
    forward = {
        "evaluable_testwindows_ge_10": metrics["evaluable_testwindow_count"] >= 10,
        "positive_testwindow_ratio_ge_65": metrics["positive_testwindow_ratio"] >= 0.65,
        "aggregate_oos_period_pf_gt_1_10": (metrics["period_profit_factor"] or 0) > 1.10,
        "20bps_stress_pf_gt_1": (metrics["20bps_stress_pf"] or 0) > 1.0,
        "30bps_stress_pf_gt_0_95": (metrics["30bps_stress_pf"] or 0) > 0.95,
        "aggregate_oos_CAGR_gt_0": (metrics["aggregate_oos_CAGR"] or 0) > 0,
        "Sharpe_gt_0": (metrics["Sharpe"] or 0) > 0,
        "maximum_drawdown_within_risk_gate": abs(metrics["maximum_drawdown"] or -1) <= 0.25,
        "single_asset_lt_40": metrics["single_asset_contribution_max"] < 0.40,
        "single_region_lt_60": metrics["single_region_contribution_max"] < 0.60,
        "single_sleeve_lt_70": metrics["single_sleeve_contribution_max"] < 0.70,
        "single_year_lt_50": metrics["single_year_contribution_max"] < 0.50,
        "PBO_lte_0_20": (metrics["PBO"] or 1) <= 0.20,
        "DSR_gt_phase6_2": (metrics["DSR_probability"] or 0) > 0.1336,
        "bootstrap_probability_return_gt_0_ge_90": (metrics["bootstrap_probability_return_gt_0"] or 0) >= 0.90,
        "bootstrap_probability_beat_diversified_champion_ge_75": (alpha_div["paired_bootstrap_probability_strategy_gt_benchmark"] or 0) >= 0.75,
        "hypothesis_mechanism_positive": ablation["component_status"] in {"ROBUSTLY_ADDS_VALUE", "MIXED_VALUE"},
        "leave_one_out_not_fragile": not any(row["fragility_status"] in {"FRAGILE", "DOMINATED_BY_ENTITY"} for row in loo),
    }
    if all(forward.values()):
        decision = "FORWARD_RESEARCH_SHADOW_ELIGIBLE"
    elif all(promising.values()):
        decision = "PROMISING_MECHANISM_CANDIDATE"
    else:
        decision = "NO_NEW_FINANCIAL_CANDIDATE"
    return _json_artifact(
        "phase6_4_decision_v1",
        {
            "technical_marker": PHASE6_4_MARKER,
            "financial_decision": decision,
            "selected_hypothesis_id": best_id,
            "promising_gates": promising,
            "forward_shadow_eligible_gates": forward,
            "FINANCIAL_FINALIST_GO": False,
            "PAPER_STRATEGY_AUTHORITY": "blocked",
            "LIVE_STRATEGY_AUTHORITY": "blocked",
            "financial_calls": _call_counters(),
            "order_calls": 0,
            "market_data_calls": 0,
            "historical_data_calls": 0,
            "account_calls": 0,
        },
        preregistration_hash=preregistration_hash,
        input_hashes={},
    )


def phase6_4_blocked_decision(preregistration_hash: str | None, reason: str) -> dict[str, Any]:
    return _json_artifact(
        "phase6_4_decision_v1",
        {
            "technical_marker": "NO_GO",
            "financial_decision": "METRIC_OR_DATA_BLOCKED",
            "blocked_reason": reason,
            "FINANCIAL_FINALIST_GO": False,
            "PAPER_STRATEGY_AUTHORITY": "blocked",
            "LIVE_STRATEGY_AUTHORITY": "blocked",
            "financial_calls": _call_counters(),
            "order_calls": 0,
            "market_data_calls": 0,
            "historical_data_calls": 0,
            "account_calls": 0,
        },
        preregistration_hash=preregistration_hash,
        input_hashes={},
    )


def phase6_4_manifest(
    project_root: Path,
    preregistration: dict[str, Any],
    reports: list[dict[str, Any]],
    benchmarks: dict[str, dict[str, Any]],
    diversified_champion: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    layout = Phase64Layout.from_project_root(project_root)
    artifacts = required_artifacts(layout)
    return _json_artifact(
        "phase6_4_manifest_v1",
        {
            "status": "GO",
            "technical_marker": PHASE6_4_MARKER,
            "financial_decision": decision["financial_decision"],
            "summary": {
                "hypothesis_count": len(reports),
                "ablation_count": len(ABLATIONS),
                "benchmark_count": len(benchmarks),
                "best_hypothesis_id": decision.get("selected_hypothesis_id"),
                "window_count": reports[0]["metrics"]["testwindow_count"] if reports else 0,
            },
            "diversified_benchmark_champion": diversified_champion["strategy_id"],
            "BUY_AND_HOLD_GLD_reference": "BUY_AND_HOLD_GLD",
            "strategy_rejection": rejected_strategy_record(project_root),
            "artifact_hashes": {str(path): sha256_file(path) for path in artifacts.values() if path.exists()},
            "call_counters": _call_counters(),
            "financial_calls": _call_counters(),
            "order_calls": 0,
            "market_data_calls": 0,
            "historical_data_calls": 0,
            "account_calls": 0,
        },
        preregistration_hash=preregistration["registry_hash"],
        input_hashes=input_hashes(project_root),
    )


def forward_shadow_spec(project_root: Path, report: dict[str, Any], preregistration_hash: str) -> dict[str, Any]:
    hypothesis = report["registry_hypothesis"] or {}
    return _json_artifact(
        "phase6_4_forward_shadow_spec_v1",
        {
            "shadow_id": "PHASE6_4_FORWARD_RESEARCH_SHADOW",
            "hypothesis_id": report["hypothesis_id"],
            "hypothesis_version": hypothesis.get("version"),
            "decision_frequency": "monthly_after_closed_daily_bars",
            "required_closed_data": "frozen EUR total-return daily dataset",
            "signal_timing": "closed-data signals only; next valid session execution proxy",
            "execution_proxy": "research_only_no_orders",
            "eligible_universe_rule": hypothesis.get("eligible_universe_rule"),
            "weighting_rule": hypothesis.get("weighting_rule"),
            "cash_rule": hypothesis.get("cash_rule"),
            "risk_constraints": hypothesis.get("risk_constraints"),
            "cost_assumption": hypothesis.get("transaction_costs"),
            "dataset_hashes": dataset_hashes(project_root),
            "parameter_hash": hypothesis.get("parameter_hash"),
            "evaluation_horizon": "forward research only",
            "minimum_forward_decisions": 12,
            "minimum_forward_days": 252,
            "authority": "NONE",
            "orders_allowed": False,
            "paper_orders_allowed": False,
            "live_orders_allowed": False,
            "status": "FORWARD_RESEARCH_SHADOW_SPEC_GO",
        },
        preregistration_hash=preregistration_hash,
        input_hashes=input_hashes(project_root),
    )


def _hypothesis_result_row(report: dict[str, Any], preregistration_hash: str) -> dict[str, Any]:
    return _parquet_row(report["metrics"], preregistration_hash, {})


def required_artifacts(layout: Phase64Layout) -> dict[str, Path]:
    return {
        "preregistration": layout.preregistration_json,
        "hypothesis_results": layout.hypothesis_results_parquet,
        "window_results": layout.window_results_parquet,
        "incremental_alpha": layout.incremental_alpha_parquet,
        "ablation_results": layout.ablation_results_parquet,
        "leave_one_out": layout.leave_one_out_parquet,
        "statistical_validation": layout.statistical_validation_json,
        "candidate_ranking": layout.candidate_ranking_json,
        "decision": layout.decision_json,
        "manifest": layout.manifest_json,
    }


def is_monthly_rebalance(dates: list[str], index: int, offset: int = 0) -> bool:
    shifted = max(0, index - offset)
    if shifted == 0:
        return True
    current = dates[shifted][:7]
    previous = dates[shifted - 1][:7]
    return current != previous


def trend_breadth(prepared: dict[str, Any], keys: list[str], index: int) -> float:
    if not keys:
        return 0.0
    positives = sum(1 for key in keys if prepared["prices"][key][index] > _mean(prepared["prices"][key][index - 200 : index]))
    return positives / len(keys)


def inverse_vol_weights(prepared: dict[str, Any], keys: list[str], index: int, window: int) -> dict[str, float]:
    if not keys or index < window:
        return {}
    inv = {key: 1.0 / max(_std(prepared["returns"][key][index - window : index]), 1e-6) for key in keys}
    total = sum(inv.values())
    return {key: value / total for key, value in inv.items()} if total else {}


def cap_weights(weights: dict[str, float], prepared: dict[str, Any], *, instrument_cap: float, region_cap: float, sleeve_cap: float, cash_key: str) -> dict[str, float]:
    capped = {key: min(value, instrument_cap) for key, value in weights.items()}
    capped = cap_group(capped, prepared, "region", region_cap)
    capped = cap_group(capped, prepared, "sleeve", sleeve_cap)
    residual = max(0.0, 1.0 - sum(capped.values()))
    capped[cash_key] = capped.get(cash_key, 0.0) + residual
    return normalize(capped)


def cap_group(weights: dict[str, float], prepared: dict[str, Any], field: str, cap: float) -> dict[str, float]:
    result = dict(weights)
    groups: dict[str, list[str]] = {}
    for key in weights:
        if key in prepared["metadata"]:
            groups.setdefault(str(prepared["metadata"][key][field]), []).append(key)
    for keys in groups.values():
        total = sum(result.get(key, 0.0) for key in keys)
        if total > cap and total > 0:
            scale = cap / total
            for key in keys:
                result[key] *= scale
    return result


def scale_weights(weights: dict[str, float], scale: float) -> dict[str, float]:
    return {key: value * scale for key, value in weights.items()}


def normalize(weights: dict[str, float]) -> dict[str, float]:
    clean = {key: value for key, value in weights.items() if value > 1e-12}
    total = sum(clean.values())
    return {key: value / total for key, value in clean.items()} if total > 1.0 else clean


def sleeve_raw_score(prepared: dict[str, Any], sleeve: str, index: int) -> float:
    keys = [key for key, meta in prepared["metadata"].items() if meta["sleeve"] == sleeve]
    if not keys:
        return -999.0
    scores = []
    for key in keys:
        prices = prepared["prices"][key]
        if prices[index - 252] <= 0 or prices[index - 126] <= 0:
            continue
        mom_12_1 = prices[index - 21] / prices[index - 252] - 1
        mom_6 = prices[index] / prices[index - 126] - 1
        trend = 1.0 if prices[index] > _mean(prices[index - 200 : index]) else 0.0
        scores.append(0.50 * mom_12_1 + 0.30 * mom_6 + 0.20 * trend)
    return _mean(scores) if scores else -999.0


def stress_pf(strategy_id: str, prepared: dict[str, Any], windows: list[dict[str, Any]], target_builder: Any, cost_bps: float) -> float | None:
    returns: list[float] = []
    for window in windows:
        subset = _slice_prepared(prepared, window["test_start"], window["test_end"])
        detailed = _detailed_portfolio(subset, target_builder(subset, int(window.get("rebalance_offset", 0))), cost_bps=cost_bps, parameters={}, name=strategy_id)
        returns.extend(row["net_return"] for row in detailed["daily"])
    return _profit_factor(returns)


def component_status(full: dict[str, Any], ablated: dict[str, Any]) -> str:
    cagr = _delta(full["aggregate_oos_CAGR"], ablated["aggregate_oos_CAGR"]) or 0.0
    sharpe = _delta(full["Sharpe"], ablated["Sharpe"]) or 0.0
    pf = _delta(full["period_profit_factor"], ablated["period_profit_factor"]) or 0.0
    if cagr > 0 and sharpe > 0 and pf > 0:
        return "ROBUSTLY_ADDS_VALUE"
    if cagr < 0 and sharpe < 0 and pf < 0:
        return "HARMFUL"
    if cagr <= 0 and sharpe <= 0:
        return "NO_INCREMENTAL_VALUE"
    return "MIXED_VALUE"


def candidate_basic_status(metrics: dict[str, Any]) -> str:
    return "PASS" if (metrics["aggregate_oos_CAGR"] or 0) > 0 and (metrics["period_profit_factor"] or 0) > 1 else "FAIL"


def concentration_blocked(metrics: dict[str, Any]) -> bool:
    return (
        metrics["single_asset_contribution_max"] > 0.40
        or metrics["single_region_contribution_max"] > 0.60
        or metrics["single_sleeve_contribution_max"] > 0.70
        or metrics["single_year_contribution_max"] > 0.50
    )


def dominance_metrics(prepared: dict[str, Any], contribution_by_asset: dict[str, float], contribution_by_year: dict[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, value) for key, value in contribution_by_asset.items()}
    total = sum(positive.values())
    region: dict[str, float] = {}
    sleeve: dict[str, float] = {}
    for key, value in positive.items():
        if key in prepared["metadata"]:
            region[prepared["metadata"][key]["region"]] = region.get(prepared["metadata"][key]["region"], 0.0) + value
            sleeve[prepared["metadata"][key]["sleeve"]] = sleeve.get(prepared["metadata"][key]["sleeve"], 0.0) + value
    region_total = sum(region.values())
    sleeve_total = sum(sleeve.values())
    years = {key: max(0.0, value) for key, value in contribution_by_year.items()}
    year_total = sum(years.values())
    return {
        "single_asset_contribution_max": max((value / total for value in positive.values()), default=0.0) if total else 0.0,
        "single_region_contribution_max": max((value / region_total for value in region.values()), default=0.0) if region_total else 0.0,
        "single_sleeve_contribution_max": max((value / sleeve_total for value in sleeve.values()), default=0.0) if sleeve_total else 0.0,
        "single_year_contribution_max": max((value / year_total for value in years.values()), default=0.0) if year_total else 0.0,
    }


def episode_pf(values: list[float]) -> dict[str, Any]:
    positives = sum(value for value in values if value > 0)
    negatives = abs(sum(value for value in values if value < 0))
    if not values:
        return {"episode_pf": None, "pf_status": "NO_CLOSED_EPISODES"}
    if positives == 0 and negatives > 0:
        return {"episode_pf": 0.0, "pf_status": "ONLY_LOSS_EPISODES"}
    if negatives == 0:
        return {"episode_pf": None, "pf_status": "NO_LOSING_EPISODES"}
    return {"episode_pf": positives / negatives, "pf_status": "EVALUABLE"}


def sortino(values: list[float]) -> float | None:
    downside = [min(0.0, value) for value in values]
    stdev = _std(downside)
    return None if stdev == 0 else _mean(values) / stdev * math.sqrt(252)


def pbo_for_report(report: dict[str, Any], reports: list[dict[str, Any]]) -> float:
    by_window: dict[str, list[float]] = {}
    for item in reports:
        for row in item["window_rows"]:
            by_window.setdefault(row["window_id"], []).append(row["window_return"])
    below = 0
    total = 0
    for row in report["window_rows"]:
        values = by_window.get(row["window_id"], [])
        if values:
            total += 1
            below += 1 if row["window_return"] < _median(values) else 0
    return below / total if total else 1.0


def paired_window_returns(left_daily: list[dict[str, Any]], right_daily: list[dict[str, Any]]) -> list[tuple[float, float]]:
    left: dict[str, list[float]] = {}
    right: dict[str, list[float]] = {}
    for row in left_daily:
        left.setdefault(row["window_id"], []).append(row["net_return"])
    for row in right_daily:
        right.setdefault(row["window_id"], []).append(row["net_return"])
    return [(_compound(left[key]), _compound(right[key])) for key in sorted(set(left) & set(right))]


def remove_entity(prepared: dict[str, Any], entity_type: str, entity: str) -> dict[str, Any]:
    metadata = prepared["metadata"]
    if entity_type == "instrument":
        keep = {key for key in metadata if key != entity}
    elif entity_type == "region":
        keep = {key for key, meta in metadata.items() if str(meta["region"]) != entity}
    elif entity_type == "sleeve":
        keep = {key for key, meta in metadata.items() if str(meta["sleeve"]) != entity}
    else:
        raise ValueError(f"unknown entity_type: {entity_type}")
    return {**prepared, "returns": {key: value for key, value in prepared["returns"].items() if key in keep}, "prices": {key: value for key, value in prepared["prices"].items() if key in keep}, "metadata": {key: value for key, value in metadata.items() if key in keep}, "cash_key": prepared["cash_key"] if prepared["cash_key"] in keep else None}


def _symbol_key(prepared: dict[str, Any], symbol: str) -> str:
    for key, meta in prepared["metadata"].items():
        if meta["symbol"] == symbol:
            return key
    raise ValueError(f"symbol not found: {symbol}")


def _optional_symbol_key(prepared: dict[str, Any], symbol: str) -> str | None:
    for key, meta in prepared["metadata"].items():
        if meta["symbol"] == symbol:
            return key
    return None


def _cash_key(prepared: dict[str, Any]) -> str:
    return str(prepared.get("cash_key") or "CASH")


def _shift_to_existing_date(dates: list[str], start: str, shift: int) -> str:
    try:
        index = dates.index(start)
    except ValueError:
        index = next(i for i, day in enumerate(dates) if day >= start)
    return dates[min(len(dates) - 1, index + shift)]


def _compound(values: list[float]) -> float:
    total = 1.0
    for value in values:
        total *= 1.0 + value
    return total - 1.0


def _navs(values: list[float]) -> list[float]:
    nav = 100.0
    navs = [nav]
    for value in values:
        nav *= 1.0 + value
        navs.append(nav)
    return navs


def _merge(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _finite(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    return output if math.isfinite(output) else 0.0


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows or [_parquet_row({"empty": True}, None, {})]), path)


def _json_artifact(
    schema: str,
    payload: dict[str, Any],
    *,
    preregistration_hash: str | None,
    input_hashes: Mapping[str, str | None],
) -> dict[str, Any]:
    base = {
        "schema": schema,
        "generated_at": utc_now_iso(),
        "input_hashes": input_hashes,
        "preregistration_hash": preregistration_hash,
        "calculation_version": CALCULATION_VERSION,
        "random_seed": PHASE6_4_RANDOM_SEED,
        **payload,
        "financial_calls": payload.get("financial_calls", _call_counters()),
        "order_calls": payload.get("order_calls", 0),
        "market_data_calls": payload.get("market_data_calls", 0),
        "historical_data_calls": payload.get("historical_data_calls", 0),
        "account_calls": payload.get("account_calls", 0),
    }
    base["content_hash"] = stable_hash({key: value for key, value in base.items() if key != "content_hash"})
    return base


def _parquet_row(
    payload: dict[str, Any],
    preregistration_hash: str | None,
    input_hashes: Mapping[str, str | None],
) -> dict[str, Any]:
    row = {
        **payload,
        "schema": payload.get("schema", "phase6_4_parquet_row_v1"),
        "generated_at": utc_now_iso(),
        "input_hashes": json.dumps(input_hashes, sort_keys=True),
        "preregistration_hash": preregistration_hash,
        "calculation_version": CALCULATION_VERSION,
        "random_seed": PHASE6_4_RANDOM_SEED,
        "financial_calls": 0,
        "order_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "account_calls": 0,
    }
    row["content_hash"] = stable_hash({key: value for key, value in row.items() if key != "content_hash"})
    return row


def _call_counters() -> dict[str, int]:
    return {
        "financial_calls": 0,
        "order_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "account_calls": 0,
    }


def input_hashes(project_root: Path) -> dict[str, str | None]:
    return {
        "phase6_3_freeze": sha256_file(Phase63Layout.from_project_root(project_root).freeze_json),
        "phase6_2_status": sha256_file(Phase62Layout.from_project_root(project_root).status_json),
        "phase6_benchmarks": sha256_file(Phase6Layout.from_project_root(project_root).benchmarks_json),
        "registry_yaml": sha256_file(Phase64Layout.from_project_root(project_root).registry_yaml),
    }


def dataset_hashes(project_root: Path) -> dict[str, str | None]:
    return {str(path): sha256_file(path) for path in sorted((project_root / "data" / "total_returns").rglob("*.parquet"))}


def phase6_3_variant_count() -> int:
    return 21
