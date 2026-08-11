from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocks.regimes.audit import (
    HMMStatePersistence,
    audit_transition_stability,
)
from stocks.regimes.features import (
    REQUIRED_DAILY_FEATURES,
    build_cross_asset_raw,
    engineer_daily_cross_asset_features,
    engineer_short_term_features,
    load_point_in_time_macro,
    standardize_train_oos,
)
from stocks.regimes.filter import hamilton_filter
from stocks.regimes.model import (
    FrozenHMM,
    fit_markov_regression,
    frozen_hmm_from_payload,
)
from stocks.regimes.risk_overlay import regime_multiplier
from stocks.research.phase11_6 import nested_walk_forward_folds
from stocks.research.phase11_8 import _run_portfolio
from stocks.research.phase11_9 import (
    BASE_STRATEGIES,
    ENSEMBLES,
    TIMEFRAMES,
    _load_current_frames,
    _load_frames,
    _parameters,
    _signals,
)


SCHEMA = "phase11_11_hmm_regime_research_v1"
MARKER = "PHASE11_11_HMM_REGIME_RESEARCH_GO"
AUTHORITY = {
    "FINANCIAL_FINALIST_GO": False,
    "STRATEGY_AUTHORITY": "NONE",
    "EXECUTION_AUTHORITY": "NONE",
    "PAPER_STRATEGY_AUTHORITY": "NONE",
    "LIVE_STRATEGY_AUTHORITY": "NONE",
    "BROKER_CALLS": 0,
    "ORDER_CALLS": 0,
}


def regimes_schema(project_root: Path) -> dict[str, Any]:
    config = _config(project_root)
    _validate_config(config)
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "configuration": config,
        "paired_strategy_count": len(BASE_STRATEGIES) + len(ENSEMBLES),
        "paired_timeframe_count": len(TIMEFRAMES),
        "paired_hypothesis_count": (
            (len(BASE_STRATEGIES) + len(ENSEMBLES)) * len(TIMEFRAMES)
        ),
        "causality": {
            "fit": "TRAIN_ONLY",
            "oos_inference": "RECURSIVE_HAMILTON_FILTER_ONLY",
            "smoother": "FORBIDDEN",
            "macro_alignment": "AVAILABLE_AT_LOCF_NO_INTERPOLATION",
            "overlay": "CAN_ONLY_DECREASE_OR_BLOCK_RISK",
        },
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "schema.json", report)
    return report


def regimes_fit(project_root: Path) -> dict[str, Any]:
    config = _config(project_root)
    frames = _load_frames(project_root)["1d"]
    features, feature_audit = _feature_matrix(project_root, frames, "1d")
    window = int(config["model"]["training_window_bars"]["1d"])
    train = features.tail(window)
    train, _, selected = _select_fold_features(train, train)
    scaled, _, scaler = standardize_train_oos(train, train)
    model = fit_markov_regression(
        scaled,
        n_regimes=int(config["model"]["regime_counts"][0]),
        search_reps=int(config["model"]["production_search_reps"]),
        max_iterations=int(config["model"]["max_iterations"]),
        variance_floor=float(config["model"]["variance_floor"]),
    )
    model = _attach_scaler(model, scaler)
    persistence = HMMStatePersistence(_private(project_root))
    stored = persistence.save_model(model)
    probabilities = _canonical_initial_probabilities(model)
    state = {
        "schema": "hmm_current_filtered_state_v1",
        "as_of": train.index[-1].isoformat(),
        "probabilities": probabilities,
        "regime_multiplier": _probability_multiplier(
            probabilities,
            config,
        ),
        "model_hash": stored["model_hash"],
        "filtered_only": True,
    }
    persistence.append_state(state)
    _write_json(_private(project_root) / "current-state.json", state)
    report = {
        "schema": "phase11_11_hmm_fit_v1",
        "status": "GO" if model.converged else "DEGRADED",
        "timeframe": "1d",
        "training_start": train.index[0],
        "training_end": train.index[-1],
        "training_observations": len(train),
        "converged": model.converged,
        "expected_durations": {
            model.raw_to_label[state_id]: model.expected_durations[state_id]
            for state_id in range(model.n_regimes)
        },
        "feature_audit": feature_audit,
        "selected_features": selected,
        "private_model_reference": stored,
        "current_state": state,
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "fit.json", report)
    return report


def regimes_walk_forward(project_root: Path) -> dict[str, Any]:
    config = _config(project_root)
    output = _output(project_root)
    phase119 = project_root / "output" / "research" / "phase11_9"
    required = (
        phase119 / "nested-results.parquet",
        phase119 / "parameter-selections.csv",
    )
    if not all(path.exists() for path in required):
        return {
            "schema": SCHEMA,
            "status": "PHASE11_9_EVIDENCE_MISSING",
            **AUTHORITY,
        }
    baseline = pd.read_parquet(required[0])
    selections = pd.read_csv(required[1])
    frames_by_timeframe = _load_frames(project_root)
    result_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    strategies = (*BASE_STRATEGIES, *ENSEMBLES)
    for timeframe in TIMEFRAMES:
        checkpoint = output / f"checkpoint-{timeframe}.parquet"
        model_checkpoint = output / f"checkpoint-models-{timeframe}.csv"
        if checkpoint.exists() and model_checkpoint.exists():
            result_rows.extend(pd.read_parquet(checkpoint).to_dict("records"))
            model_rows.extend(pd.read_csv(model_checkpoint).to_dict("records"))
            continue
        frames = frames_by_timeframe.get(timeframe, {})
        if len(frames) < 5:
            blocked.append(
                {
                    "timeframe": timeframe,
                    "reason": "INSUFFICIENT_REAL_ASSETS",
                }
            )
            continue
        features, feature_audit = _feature_matrix(
            project_root,
            frames,
            timeframe,
        )
        start = min(frame.index.min() for frame in frames.values())
        end = min(frame.index.max() for frame in frames.values())
        folds = nested_walk_forward_folds(start, end, timeframe)
        signal_cache: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
        timeframe_rows: list[dict[str, Any]] = []
        timeframe_models: list[dict[str, Any]] = []
        for fold in folds.to_dict("records"):
            fold_id = str(fold["fold_id"])
            train = features.loc[
                (features.index >= pd.Timestamp(fold["train_start"]))
                & (features.index <= pd.Timestamp(fold["train_end"]))
            ]
            window = int(
                config["model"]["training_window_bars"][timeframe]
            )
            train = train.tail(window)
            oos = features.loc[
                (features.index > pd.Timestamp(fold["train_end"]))
                & (features.index <= pd.Timestamp(fold["outer_test_end"]))
            ]
            train, oos, selected_features = _select_fold_features(
                train,
                oos,
            )
            if len(train) < 100 or oos.empty:
                blocked.append(
                    {
                        "timeframe": timeframe,
                        "fold_id": fold_id,
                        "reason": "INSUFFICIENT_HMM_FEATURE_ROWS",
                    }
                )
                continue
            scaled_train, scaled_oos, scaler = standardize_train_oos(
                train,
                oos,
            )
            try:
                model = fit_markov_regression(
                    scaled_train,
                    n_regimes=int(config["model"]["regime_counts"][0]),
                    search_reps=int(config["model"]["search_reps"]),
                    max_iterations=int(config["model"]["max_iterations"]),
                    variance_floor=float(config["model"]["variance_floor"]),
                )
            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                blocked.append(
                    {
                        "timeframe": timeframe,
                        "fold_id": fold_id,
                        "reason": f"HMM_FIT_BLOCKED:{type(exc).__name__}",
                    }
                )
                continue
            model = _attach_scaler(model, scaler)
            probabilities = hamilton_filter(model, scaled_oos)
            stability = audit_transition_stability(
                model,
                probabilities,
                minimum_state_fraction=float(
                    config["model"]["minimum_state_observation_fraction"]
                ),
                minimum_duration=float(
                    config["model"]["minimum_expected_duration_bars"]
                ),
                maximum_chatter_ratio=float(
                    config["model"]["maximum_single_bar_chatter_ratio"]
                ),
            )
            multiplier = regime_multiplier(
                probabilities,
                _state_multipliers(config),
                minimum=float(
                    config["risk_bounds"]["minimum_exposure_multiplier"]
                ),
                maximum=float(
                    config["risk_bounds"]["maximum_exposure_multiplier"]
                ),
            )
            timeframe_models.append(
                {
                    "timeframe": timeframe,
                    "fold_id": fold_id,
                    "converged": model.converged,
                    "stability_status": stability["status"],
                    "training_observations": len(train),
                    "oos_probability_rows": len(probabilities),
                    "minimum_multiplier": float(multiplier.min()),
                    "mean_multiplier": float(multiplier.mean()),
                    "feature_count": len(features.columns),
                    "selected_feature_count": len(selected_features),
                    "selected_features": ",".join(selected_features),
                    "selected_macro_features": ",".join(
                        sorted(
                            set(selected_features)
                            - set(REQUIRED_DAILY_FEATURES)
                        )
                    )
                    if timeframe not in {"1h", "4h"}
                    else "",
                    "macro_features_used": ",".join(
                        feature_audit["macro_features_used"]
                    ),
                }
            )
            for strategy in strategies:
                selection = selections.loc[
                    selections["timeframe"].eq(timeframe)
                    & selections["strategy"].eq(strategy)
                    & selections["fold_id"].eq(fold_id)
                ]
                if selection.empty:
                    blocked.append(
                        {
                            "timeframe": timeframe,
                            "fold_id": fold_id,
                            "strategy": strategy,
                            "reason": "PHASE11_9_PROFILE_SELECTION_MISSING",
                        }
                    )
                    continue
                profile = str(selection.iloc[0]["selected_profile"])
                cache_key = strategy, profile
                if cache_key not in signal_cache:
                    signal_cache[cache_key] = _signals(
                        frames,
                        strategy,
                        timeframe,
                        profile,
                    )
                for cost_bps in (10.0, 50.0):
                    base_row = baseline.loc[
                        baseline["timeframe"].eq(timeframe)
                        & baseline["strategy"].eq(strategy)
                        & baseline["fold_id"].eq(fold_id)
                        & baseline["cost_bps"].eq(cost_bps)
                    ]
                    if base_row.empty:
                        continue
                    hmm_run = _run_portfolio(
                        frames,
                        signal_cache[cache_key],
                        start=pd.Timestamp(fold["outer_test_start"]),
                        end=pd.Timestamp(fold["outer_test_end"]),
                        cost_bps=cost_bps,
                        exposure_multiplier=multiplier,
                        entry_block_below_multiplier=float(
                            config["risk_bounds"][
                                "entry_block_below_multiplier"
                            ]
                        ),
                    )
                    base = base_row.iloc[0]
                    metrics = hmm_run["metrics"]
                    timeframe_rows.append(
                        {
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "fold_id": fold_id,
                            "profile": profile,
                            "cost_bps": cost_bps,
                            "model_stability_status": stability["status"],
                            "baseline_CAGR": base["CAGR"],
                            "hmm_CAGR": metrics["CAGR"],
                            "baseline_Sharpe": base["Sharpe"],
                            "hmm_Sharpe": metrics["Sharpe"],
                            "baseline_maximum_drawdown": base[
                                "maximum_drawdown"
                            ],
                            "hmm_maximum_drawdown": metrics[
                                "maximum_drawdown"
                            ],
                            "baseline_period_profit_factor": base[
                                "period_profit_factor"
                            ],
                            "hmm_period_profit_factor": metrics[
                                "period_profit_factor"
                            ],
                            "baseline_fill_count": base["fill_count"],
                            "hmm_fill_count": len(hmm_run["fills"]),
                            "hmm_mean_exposure_multiplier": float(
                                multiplier.mean()
                            ),
                        }
                    )
        result_rows.extend(timeframe_rows)
        model_rows.extend(timeframe_models)
        _write_frame(checkpoint, pd.DataFrame(timeframe_rows))
        _write_frame(model_checkpoint, pd.DataFrame(timeframe_models))
    results = pd.DataFrame(result_rows)
    models = pd.DataFrame(model_rows)
    summary = _paired_summary(results)
    promotions = _promotion_registry(summary, config)
    frozen_shadow = _frozen_shadow_registry(promotions, selections)
    _write_frame(output / "paired-results.parquet", results)
    _write_frame(output / "paired-summary.csv", summary)
    _write_frame(output / "model-fold-audit.csv", models)
    _write_json(output / "promotion-registry.json", promotions)
    _write_json(output / "frozen-shadow-registry.json", frozen_shadow)
    _write_json(output / "blocked.json", blocked)
    report = {
        "schema": SCHEMA,
        "status": "GO" if not results.empty else "NO_GO",
        "phase11_11_marker": (
            MARKER
            if not results.empty
            else "PHASE11_11_HMM_REGIME_RESEARCH_NO_GO"
        ),
        "strategy_count": len(strategies),
        "timeframe_count": len(TIMEFRAMES),
        "strategy_timeframe_pairs_expected": (
            len(strategies) * len(TIMEFRAMES)
        ),
        "strategy_timeframe_pairs_evaluated": int(
            summary[["strategy", "timeframe"]].drop_duplicates().shape[0]
        )
        if not summary.empty
        else 0,
        "paired_result_count": len(results),
        "hmm_model_fold_count": len(models),
        "promotions": promotions,
        "frozen_shadow_registry": {
            "candidate_count": frozen_shadow["candidate_count"],
            "authority": frozen_shadow["execution_authority"],
        },
        "filtered_probabilities_only": True,
        "smoothed_probabilities_used": False,
        "selection_bias_status": "BLOCKED_NO_NEW_INDEPENDENT_HOLDOUT",
        **AUTHORITY,
    }
    _write_json(output / "status.json", report)
    _write_json(output / "manifest.json", _manifest(project_root, report))
    return report


def regimes_current(project_root: Path) -> dict[str, Any]:
    pointer = _private(project_root) / "current-model.json"
    state_path = _private(project_root) / "current-state.json"
    if not pointer.exists() or not state_path.exists():
        return {
            "schema": "hmm_current_regime_v1",
            "status": "MODEL_NOT_FIT",
            **AUTHORITY,
        }
    reference = json.loads(pointer.read_text(encoding="utf-8"))
    model = frozen_hmm_from_payload(
        json.loads(Path(reference["model_path"]).read_text(encoding="utf-8"))
    )
    previous = json.loads(state_path.read_text(encoding="utf-8"))
    frames = _load_current_frames(project_root)["1d"]
    features, feature_audit = _feature_matrix(project_root, frames, "1d")
    new = features.loc[features.index > pd.Timestamp(previous["as_of"])]
    if new.empty:
        report = {
            "schema": "hmm_current_regime_v1",
            "status": "GO",
            "state": previous,
            "new_filtered_rows": 0,
            "feature_audit": feature_audit,
            **AUTHORITY,
        }
        _write_json(_output(project_root) / "current.json", report)
        return report
    scaled = _scale_with_model(new, model)
    initial = _raw_probabilities(previous["probabilities"], model)
    probabilities = hamilton_filter(
        model,
        scaled,
        initial_probabilities=initial,
    )
    latest = probabilities.iloc[-1].to_dict()
    state = {
        "schema": "hmm_current_filtered_state_v1",
        "as_of": probabilities.index[-1].isoformat(),
        "probabilities": latest,
        "regime_multiplier": _probability_multiplier(
            latest,
            _config(project_root),
        ),
        "model_hash": reference["model_hash"],
        "filtered_only": True,
    }
    HMMStatePersistence(_private(project_root)).append_state(state)
    _write_json(state_path, state)
    report = {
        "schema": "hmm_current_regime_v1",
        "status": "GO",
        "state": state,
        "new_filtered_rows": len(probabilities),
        "feature_audit": feature_audit,
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "current.json", report)
    return report


def regimes_audit(project_root: Path) -> dict[str, Any]:
    status = regimes_status(project_root)
    source_root = project_root / "src" / "stocks" / "regimes"
    smoother_hits = []
    forbidden_calls = []
    for path in sorted(source_root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        smoother_api = "smoothed" + "_marginal_probabilities"
        if smoother_api in source:
            smoother_hits.append(str(path))
        for token in (
            "place" + "Order",
            "cancel" + "Order",
            "reqGlobal" + "Cancel",
            "req" + "Ids",
        ):
            if token in source:
                forbidden_calls.append({"path": str(path), "token": token})
    checks = {
        "status_available": status.get("status") == "GO",
        "all_strategy_timeframe_pairs": (
            status.get("strategy_timeframe_pairs_evaluated")
            == status.get("strategy_timeframe_pairs_expected")
        ),
        "filtered_only": status.get("filtered_probabilities_only") is True,
        "smoother_absent": not smoother_hits,
        "broker_calls_absent": not forbidden_calls,
        "authority_none": status.get("EXECUTION_AUTHORITY") == "NONE",
    }
    report = {
        "schema": "phase11_11_hmm_audit_v1",
        "status": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "smoother_hits": smoother_hits,
        "forbidden_call_hits": forbidden_calls,
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "audit.json", report)
    return report


def regimes_status(project_root: Path) -> dict[str, Any]:
    path = _output(project_root) / "status.json"
    if not path.exists():
        return {"schema": SCHEMA, "status": "NOT_RUN", **AUTHORITY}
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_matrix(
    project_root: Path,
    frames: Mapping[str, pd.DataFrame],
    timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if timeframe in {"1h", "4h"}:
        features = engineer_short_term_features(frames["SPY"])
        return features, {
            "feature_set": "SHORT_TERM",
            "macro_features_used": [],
            "point_in_time": True,
        }
    raw_index = build_cross_asset_raw(frames).index
    macro = load_point_in_time_macro(project_root, raw_index)
    raw = build_cross_asset_raw(frames, macro)
    periods_per_year = {"1d": 252.0, "1w": 52.0, "1mo": 12.0}[
        timeframe
    ]
    features = engineer_daily_cross_asset_features(
        raw,
        periods_per_year=periods_per_year,
    )
    used = sorted(
        set(macro.columns)
        & {
            "USD_INDEX",
            "US_HIGH_YIELD_SPREAD",
            "US_YIELD_CURVE_10Y2Y",
            "US_FINANCIAL_CONDITIONS",
            "VIX",
        }
    )
    return features, {
        "feature_set": "DAILY_CROSS_ASSET_MACRO",
        "macro_features_used": used,
        "point_in_time": True,
        "alignment": "AVAILABLE_AT_LOCF_NO_INTERPOLATION",
    }


def _paired_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    normal = results.loc[results["cost_bps"].eq(10.0)]
    for (strategy, timeframe), group in normal.groupby(
        ["strategy", "timeframe"]
    ):
        stress = results.loc[
            results["strategy"].eq(strategy)
            & results["timeframe"].eq(timeframe)
            & results["cost_bps"].eq(50.0)
        ]
        baseline_sharpe = _median(group["baseline_Sharpe"])
        hmm_sharpe = _median(group["hmm_Sharpe"])
        baseline_dd = float(group["baseline_maximum_drawdown"].min())
        hmm_dd = float(group["hmm_maximum_drawdown"].min())
        drawdown_reduction = (
            (abs(baseline_dd) - abs(hmm_dd)) / abs(baseline_dd)
            if baseline_dd != 0
            else math.nan
        )
        rows.append(
            {
                "strategy": strategy,
                "timeframe": timeframe,
                "fold_count": len(group),
                "baseline_median_pf": _median(
                    group["baseline_period_profit_factor"]
                ),
                "hmm_median_pf": _median(
                    group["hmm_period_profit_factor"]
                ),
                "baseline_median_CAGR": _median(group["baseline_CAGR"]),
                "hmm_median_CAGR": _median(group["hmm_CAGR"]),
                "baseline_median_Sharpe": baseline_sharpe,
                "hmm_median_Sharpe": hmm_sharpe,
                "sharpe_ablation_ratio": (
                    hmm_sharpe / baseline_sharpe
                    if baseline_sharpe > 0
                    else math.nan
                ),
                "baseline_worst_drawdown": baseline_dd,
                "hmm_worst_drawdown": hmm_dd,
                "drawdown_reduction_fraction": drawdown_reduction,
                "hmm_positive_fold_ratio": float(
                    group["hmm_CAGR"].gt(0).mean()
                ),
                "hmm_incremental_pf_fold_ratio": float(
                    group["hmm_period_profit_factor"]
                    .gt(group["baseline_period_profit_factor"])
                    .mean()
                ),
                "hmm_incremental_sharpe_fold_ratio": float(
                    group["hmm_Sharpe"].gt(group["baseline_Sharpe"]).mean()
                ),
                "baseline_cost_50bps_median_pf": _median(
                    stress["baseline_period_profit_factor"]
                ),
                "hmm_cost_50bps_median_pf": _median(
                    stress["hmm_period_profit_factor"]
                ),
                "stable_model_fold_ratio": float(
                    group["model_stability_status"].eq("GO").mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "hmm_median_pf",
            "sharpe_ablation_ratio",
            "drawdown_reduction_fraction",
        ],
        ascending=False,
    )


def _select_fold_features(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    *,
    minimum_observations: int = 100,
    minimum_optional_coverage: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if train.empty or oos.empty:
        return train.iloc[0:0], oos.iloc[0:0], []
    required = [
        column
        for column in REQUIRED_DAILY_FEATURES
        if column in train.columns
    ]
    if "world_index_ret" not in required:
        required = list(train.columns)
    base_train = train.dropna(subset=required)
    base_oos = oos.dropna(subset=required)
    if len(base_train) < minimum_observations or base_oos.empty:
        return base_train.iloc[0:0], base_oos.iloc[0:0], required
    optional = []
    for column in train.columns:
        if column in required:
            continue
        train_coverage = float(base_train[column].notna().mean())
        oos_coverage = float(base_oos[column].notna().mean())
        if (
            train_coverage >= minimum_optional_coverage
            and oos_coverage >= minimum_optional_coverage
        ):
            optional.append(column)
    selected = required + optional
    return (
        base_train.loc[:, selected].dropna(),
        base_oos.loc[:, selected].dropna(),
        selected,
    )


def _promotion_registry(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config["promotion"]
    rows = []
    for candidate in summary.to_dict("records"):
        baseline_checks = {
            "fold_count": candidate["fold_count"]
            >= thresholds["minimum_fold_count"],
            "median_pf": candidate["baseline_median_pf"]
            > thresholds["minimum_baseline_median_pf"],
            "median_cagr": candidate["baseline_median_CAGR"] > 0,
            "cost_50bps": candidate["baseline_cost_50bps_median_pf"]
            > thresholds["minimum_cost_50bps_pf"],
            "drawdown": candidate["baseline_worst_drawdown"] > -0.50,
        }
        hmm_checks = {
            "median_pf": candidate["hmm_median_pf"]
            > thresholds["minimum_hmm_median_pf"],
            "median_cagr": candidate["hmm_median_CAGR"] > 0,
            "positive_folds": candidate["hmm_positive_fold_ratio"]
            >= thresholds["minimum_hmm_positive_fold_ratio"],
            "cost_50bps": candidate["hmm_cost_50bps_median_pf"]
            > thresholds["minimum_cost_50bps_pf"],
            "sharpe_ablation": _finite(candidate["sharpe_ablation_ratio"])
            >= thresholds["minimum_sharpe_ablation_ratio"],
            "drawdown_reduction": _finite(
                candidate["drawdown_reduction_fraction"]
            )
            >= thresholds["minimum_drawdown_reduction_fraction"],
            "incremental_folds": candidate["hmm_incremental_pf_fold_ratio"]
            >= thresholds["minimum_incremental_fold_ratio"],
            "stable_model_folds": candidate["stable_model_fold_ratio"]
            >= thresholds["minimum_incremental_fold_ratio"],
        }
        if all(baseline_checks.values()) and all(hmm_checks.values()):
            decision = "HMM_OVERLAY_PROMOTED"
            selected_variant = "HMM"
            destination = "FROZEN_SHADOW"
        elif all(baseline_checks.values()):
            decision = "BASELINE_RETAINED_HMM_REJECTED"
            selected_variant = "BASELINE"
            destination = "FROZEN_SHADOW"
        else:
            decision = "NO_PROMOTION"
            selected_variant = "NONE"
            destination = "RESEARCH_CANDIDATE"
        rows.append(
            {
                "strategy": candidate["strategy"],
                "timeframe": candidate["timeframe"],
                "decision": decision,
                "selected_variant": selected_variant,
                "destination": destination,
                "baseline_checks": baseline_checks,
                "hmm_checks": hmm_checks,
            }
        )
    return {
        "schema": "phase11_11_hmm_promotion_registry_v1",
        "status": "GO",
        "strategy_timeframe_count": len(rows),
        "hmm_overlay_promoted_count": sum(
            row["decision"] == "HMM_OVERLAY_PROMOTED" for row in rows
        ),
        "baseline_retained_count": sum(
            row["decision"] == "BASELINE_RETAINED_HMM_REJECTED"
            for row in rows
        ),
        "no_promotion_count": sum(
            row["decision"] == "NO_PROMOTION" for row in rows
        ),
        "promotions": rows,
        "automatic_execution_promotion": False,
        "destination_maximum": "FROZEN_SHADOW",
        **AUTHORITY,
    }


def _frozen_shadow_registry(
    promotions: Mapping[str, Any],
    selections: pd.DataFrame,
) -> dict[str, Any]:
    candidates = []
    for row in promotions["promotions"]:
        if row["destination"] != "FROZEN_SHADOW":
            continue
        strategy = str(row["strategy"])
        timeframe = str(row["timeframe"])
        selected = selections.loc[
            selections["strategy"].eq(strategy)
            & selections["timeframe"].eq(timeframe)
        ].sort_values("fold_id")
        if selected.empty:
            continue
        profile = str(selected.iloc[-1]["selected_profile"])
        parameters = _parameters(timeframe, profile)
        candidate_core = {
            "strategy_name": strategy,
            "family": strategy,
            "timeframe": timeframe,
            "profile": profile,
            "parameters": parameters,
            "selected_variant": row["selected_variant"],
            "classification": "FROZEN_SHADOW",
            "source": "PHASE11_11_PAIRED_HMM_ABLATION",
        }
        candidates.append(
            {
                "candidate_id": "HMM11-"
                + content_hash(candidate_core)[:20],
                **candidate_core,
                "economic_interest": True,
                "automatic_authority": "NONE",
                "execution_authority": "NONE",
                "paper_strategy_authority": "NONE",
                "live_strategy_authority": "NONE",
                "limitations": [
                    "NO_NEW_INDEPENDENT_FORWARD_HOLDOUT",
                    "OBSERVATION_ONLY",
                ],
            }
        )
    return {
        "schema": "phase11_11_frozen_shadow_registry_v1",
        "status": "GO",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "automatic_activation": False,
        "signal_authority": "SHADOW",
        "execution_authority": "NONE",
        **AUTHORITY,
    }


def _attach_scaler(
    model: FrozenHMM,
    scaler: Mapping[str, Mapping[str, float]],
) -> FrozenHMM:
    return replace(
        model,
        feature_means={
            column: float(values["mean"])
            for column, values in scaler.items()
        },
        feature_scales={
            column: float(values["scale"])
            for column, values in scaler.items()
        },
    )


def _scale_with_model(
    features: pd.DataFrame,
    model: FrozenHMM,
) -> pd.DataFrame:
    mean = pd.Series(model.feature_means)
    scale = pd.Series(model.feature_scales)
    return features.loc[:, mean.index].sub(mean).div(scale)


def _canonical_initial_probabilities(
    model: FrozenHMM,
) -> dict[str, float]:
    return {
        model.raw_to_label[state]: float(model.initial_probabilities[state])
        for state in range(model.n_regimes)
    }


def _raw_probabilities(
    canonical: Mapping[str, float],
    model: FrozenHMM,
) -> np.ndarray:
    return np.array(
        [
            float(canonical[model.raw_to_label[state]])
            for state in range(model.n_regimes)
        ]
    )


def _probability_multiplier(
    probabilities: Mapping[str, float],
    config: Mapping[str, Any],
) -> float:
    multipliers = _state_multipliers(config)
    return float(
        sum(
            float(probability) * multipliers[state]
            for state, probability in probabilities.items()
        )
    )


def _state_multipliers(config: Mapping[str, Any]) -> dict[str, float]:
    return {
        state: float(settings["multiplier"])
        for state, settings in config["states"].items()
    }


def _config(project_root: Path) -> dict[str, Any]:
    return json.loads(
        (project_root / "config" / "regimes" / "hmm_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "hmm_regime_configuration_v1":
        raise ValueError("INVALID_HMM_CONFIGURATION_SCHEMA")
    multipliers = _state_multipliers(config)
    if not all(0 <= value <= 1 for value in multipliers.values()):
        raise ValueError("HMM_MULTIPLIER_OUT_OF_BOUNDS")
    if config["authority"]["execution_authority"] != "NONE":
        raise ValueError("HMM_EXECUTION_AUTHORITY_FORBIDDEN")


def _median(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    if clean.empty:
        return math.nan
    return float(clean.median())


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return number if math.isfinite(number) else -math.inf


def _private(project_root: Path) -> Path:
    return project_root / "data" / "regimes" / "private"


def _output(project_root: Path) -> Path:
    return project_root / "output" / "research" / "phase11_11"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def content_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest().upper()


def _manifest(
    project_root: Path,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    output = _output(project_root)
    artifact_names = (
        "schema.json",
        "fit.json",
        "current.json",
        "paired-results.parquet",
        "paired-summary.csv",
        "model-fold-audit.csv",
        "promotion-registry.json",
        "frozen-shadow-registry.json",
        "blocked.json",
        "audit.json",
        "status.json",
    )
    hashes = {}
    for name in artifact_names:
        path = output / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    source_paths = [
        project_root / "main.py",
        project_root / "config" / "regimes" / "hmm_v1.json",
        project_root / "src" / "stocks" / "research" / "phase11_8.py",
        project_root / "src" / "stocks" / "dynamic" / "service.py",
        project_root / "src" / "stocks" / "signals" / "service.py",
        project_root / "src" / "stocks" / "operations" / "service.py",
        project_root / "tests" / "test_hmm_regimes.py",
        project_root / "PHASE11_11_STATUS.md",
        project_root
        / "docs"
        / "PHASE11_11_HMM_REGIME_RESEARCH.md",
        *sorted(
            (project_root / "src" / "stocks" / "regimes").glob("*.py")
        ),
    ]
    return {
        "schema": "phase11_11_hmm_manifest_v1",
        "status": "GO",
        "phase11_11_marker": status.get("phase11_11_marker"),
        "artifact_hashes": hashes,
        "artifact_count": len(hashes),
        "source_hashes": {
            str(path.relative_to(project_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest().upper()
            for path in source_paths
            if path.exists()
        },
        **AUTHORITY,
    }
