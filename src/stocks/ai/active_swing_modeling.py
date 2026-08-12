from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stocks.ai.active_swing_panel import (
    CONTRACT_COLUMNS,
    INTERACTION_FEATURE_COLUMNS,
    SCALAR_FEATURE_COLUMNS,
    TIMEFRAME_FEATURE_COLUMNS,
)
from stocks.execution.idempotency import stable_hash


TOURNAMENT_PATH = Path(
    "output/ai/decision-intelligence/active-swing-tournament.json"
)
OOS_PATH = Path(
    "output/ai/decision-intelligence/active-swing-oos-predictions.parquet"
)
MODEL_PATH = Path(
    "output/ai/decision-intelligence/active-swing-shadow-model.joblib"
)
NUMERIC_FEATURES = (
    *SCALAR_FEATURE_COLUMNS,
    *(
        name
        for name in TIMEFRAME_FEATURE_COLUMNS
        if not name.startswith("trend_state_")
    ),
)
CATEGORICAL_FEATURES = (
    *CONTRACT_COLUMNS,
    *(
        name
        for name in TIMEFRAME_FEATURE_COLUMNS
        if name.startswith("trend_state_")
    ),
    "symbol",
    "strategy_id",
    "strategy_family",
    "strategy_dna_hash",
)
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
ABLATION_VARIANTS = (
    ("15M_ONLY", ("15m",)),
    ("15M_PLUS_1H", ("15m", "1h")),
    ("15M_PLUS_1H_PLUS_4H", ("15m", "1h", "4h")),
    ("15M_PLUS_1H_PLUS_4H_PLUS_1D", ("15m", "1h", "4h", "1d")),
    ("FULL_15M_1H_2H_4H_1D", ("15m", "1h", "2h", "4h", "1d")),
)


def run_active_swing_model_tournament(
    panel: pd.DataFrame,
    panel_status: Mapping[str, Any],
    *,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any] | None]:
    """Fit candidate-conditioned challengers only after fixed evidence minima."""

    blockers = list(panel_status.get("training_blockers") or [])
    if panel.empty or panel_status.get("training_ready") is not True:
        report = _blocked_report(panel, blockers)
        report["timeframe_ablations"] = _blocked_timeframe_ablations(
            panel, blockers
        )
        report["interaction_trials"] = _blocked_interaction_trials(
            panel, blockers
        )
        report["content_hash"] = stable_hash(report)
        return report, pd.DataFrame(), None
    frame = _validated_panel(panel)
    dates = pd.DatetimeIndex(
        sorted(frame["decision_timestamp"].dt.normalize().unique())
    )
    test_start = dates[max(1, int(len(dates) * 0.80))]
    training = frame.loc[
        (frame["decision_timestamp"] < test_start)
        & (frame["label_available_at"] < test_start)
    ]
    test = frame.loc[frame["decision_timestamp"] >= test_start]
    split_blockers: list[str] = []
    if len(training) < 300:
        split_blockers.append("PURGED_TRAIN_ROWS_BELOW_300")
    if len(test) < 100:
        split_blockers.append("HOLDOUT_ROWS_BELOW_100")
    if training["positive_net_trade"].nunique() < 2:
        split_blockers.append("PURGED_TRAIN_SINGLE_CLASS")
    if test["positive_net_trade"].nunique() < 2:
        split_blockers.append("HOLDOUT_SINGLE_CLASS")
    if split_blockers:
        report = _blocked_report(frame, split_blockers)
        report["status"] = "INSUFFICIENT_PURGED_HOLDOUT_EVIDENCE"
        report["purged_training_rows"] = len(training)
        report["holdout_rows"] = len(test)
        report["timeframe_ablations"] = _blocked_timeframe_ablations(
            frame, split_blockers
        )
        report["interaction_trials"] = _blocked_interaction_trials(
            frame, split_blockers
        )
        report["content_hash"] = stable_hash(report)
        return report, pd.DataFrame(), None

    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    fitted: dict[str, tuple[Pipeline, Pipeline]] = {}
    baseline_probability = float(training["positive_net_trade"].mean())
    baseline_return = float(training["native_exit_net_return"].mean())
    for family in ("LINEAR", "EXTRA_TREES"):
        classifier, regressor = _pipelines(family, random_state=random_state)
        classifier.fit(training.loc[:, FEATURES], training["positive_net_trade"])
        regressor.fit(training.loc[:, FEATURES], training["native_exit_net_return"])
        probability = classifier.predict_proba(test.loc[:, FEATURES])[:, 1]
        expected = regressor.predict(test.loc[:, FEATURES])
        realized = test["native_exit_net_return"].to_numpy(dtype=float)
        labels = test["positive_net_trade"].to_numpy(dtype=int)
        metrics = _metrics(
            labels,
            realized,
            probability,
            expected,
            baseline_probability=baseline_probability,
            baseline_return=baseline_return,
        )
        results.append({"family": family, **metrics})
        prediction = test.loc[
            :,
            [
                "candidate_identity",
                "decision_timestamp",
                "label_available_at",
                "native_exit_net_return",
                "positive_net_trade",
            ],
        ].copy()
        prediction["family"] = family
        prediction["probability_positive_net"] = probability
        prediction["predicted_net_R"] = expected
        predictions.append(prediction)
        fitted[family] = (classifier, regressor)

    selected = max(
        results,
        key=lambda row: (
            row["top_quartile_delta_net_R"],
            row["regression_spearman"],
            row["roc_auc"],
        ),
    )
    selected_family = str(selected["family"])
    timeframe_ablations = _run_timeframe_ablations(
        training,
        test,
        family=selected_family,
        random_state=random_state,
        baseline_probability=baseline_probability,
        baseline_return=baseline_return,
    )
    interaction_trials = _run_interaction_trials(
        training,
        test,
        family=selected_family,
        random_state=random_state,
        baseline_probability=baseline_probability,
        baseline_return=baseline_return,
        timeframe_ablations=timeframe_ablations,
    )
    final_classifier, final_regressor = _pipelines(
        selected_family, random_state=random_state
    )
    final_classifier.fit(frame.loc[:, FEATURES], frame["positive_net_trade"])
    final_regressor.fit(frame.loc[:, FEATURES], frame["native_exit_net_return"])
    expected_win = float(
        frame.loc[frame["native_exit_net_return"] > 0, "native_exit_net_return"].mean()
    )
    expected_loss = float(
        frame.loc[frame["native_exit_net_return"] <= 0, "native_exit_net_return"].mean()
    )
    model_version = stable_hash(
        {
            "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
            "panel_hash": panel_status.get("panel_sha256"),
            "family": selected_family,
            "rows": len(frame),
            "test_start": test_start.isoformat(),
        }
    )[:24]
    bundle = {
        "schema": "active_swing_candidate_shadow_model_bundle_v1",
        "model_version": model_version,
        "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
        "prediction_unit": "ONE_CANDIDATE_PER_PREDICTION",
        "classifier_family": selected_family,
        "regressor_family": selected_family,
        "classifier": final_classifier,
        "regressor": final_regressor,
        "feature_columns": list(FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "training_rows": len(frame),
        "training_candidate_identities": sorted(
            frame["candidate_identity"].astype(str).unique()
        ),
        "expected_win_R": expected_win,
        "expected_loss_R": expected_loss,
        "holdout_start": test_start.isoformat(),
        "authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
        "broker_writes": 0,
    }
    oos = pd.concat(predictions, ignore_index=True)
    report: dict[str, Any] = {
        "schema": "active_swing_candidate_model_tournament_v1",
        "status": "SHADOW_MODEL_TRAINED_NOT_PROMOTED",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
        "prediction_unit": "ONE_CANDIDATE_PER_PREDICTION",
        "training_rows": len(training),
        "holdout_rows": len(test),
        "holdout_start": test_start.isoformat(),
        "label_availability_purged": True,
        "candidate_identity_deduplicated": True,
        "challengers": results,
        "selected_family": selected_family,
        "timeframe_ablations": timeframe_ablations,
        "timeframe_architecture_trial_count": len(timeframe_ablations),
        "interaction_trials": interaction_trials,
        "cross_timeframe_interaction_trial_count": len(interaction_trials),
        "cross_timeframe_interactions_used_by_final_model": [],
        "interaction_selection_policy": (
            "REGISTER_AND_EVALUATE_SEPARATELY_NO_HOLDOUT_SELECTION_IN_FINAL_MODEL"
        ),
        "model_version": model_version,
        "promotion_status": "NOT_ELIGIBLE_FORWARD_AND_EXTERNAL_GATES_REQUIRED",
        "performance_gate_go": False,
        "external_data_gate_go": False,
        "forward_evidence_go": False,
        "automatic_promotion": False,
        "financial_authority": False,
        "authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }
    report["content_hash"] = stable_hash(report)
    bundle["tournament_hash"] = report["content_hash"]
    return report, oos, bundle


def predict_active_swing_bundle(
    bundle: Mapping[str, Any], features: pd.DataFrame
) -> pd.DataFrame:
    columns = list(bundle["feature_columns"])
    probability = bundle["classifier"].predict_proba(features.loc[:, columns])[:, 1]
    expected = bundle["regressor"].predict(features.loc[:, columns])
    conservative = (
        probability * float(bundle["expected_win_R"])
        + (1.0 - probability) * float(bundle["expected_loss_R"])
    )
    uncertainty = np.clip(
        -(probability * np.log(probability.clip(1e-12))
          + (1.0 - probability) * np.log((1.0 - probability).clip(1e-12)))
        / math.log(2),
        0.0,
        1.0,
    )
    return pd.DataFrame(
        {
            "probability_positive_net": probability,
            "predicted_net_R": expected,
            "conservative_expected_R": conservative,
            "uncertainty": uncertainty,
            "abstained": (np.abs(probability - 0.5) < 0.10) | (uncertainty > 0.90),
        },
        index=features.index,
    )


def _validated_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        *FEATURES,
        "candidate_identity",
        "candidate_unit",
        "decision_timestamp",
        "label_available_at",
        "positive_net_trade",
        "native_exit_net_return",
    }
    if not required.issubset(panel.columns):
        raise ValueError(
            f"active-swing panel missing columns: {sorted(required - set(panel))}"
        )
    frame = panel.copy()
    frame["decision_timestamp"] = pd.to_datetime(
        frame["decision_timestamp"], utc=True, errors="coerce"
    )
    frame["label_available_at"] = pd.to_datetime(
        frame["label_available_at"], utc=True, errors="coerce"
    )
    frame = frame.dropna(
        subset=["decision_timestamp", "label_available_at", "native_exit_net_return"]
    ).sort_values("decision_timestamp")
    if frame["candidate_identity"].duplicated().any():
        raise ValueError("active-swing training panel contains duplicate candidates")
    if not frame["candidate_unit"].eq("ONE_NATURAL_STRATEGY_SETUP").all():
        raise ValueError("active-swing training panel has an incompatible unit")
    return frame.reset_index(drop=True)


def _pipelines(
    family: str,
    *,
    random_state: int,
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> tuple[Pipeline, Pipeline]:
    if family == "LINEAR":
        classifier_model = LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=2_000, random_state=random_state
        )
        regressor_model = Ridge(alpha=1.0)
    elif family == "EXTRA_TREES":
        classifier_model = ExtraTreesClassifier(
            n_estimators=128,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
        regressor_model = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError(f"unsupported active-swing challenger: {family}")
    return (
        Pipeline(
            [
                (
                    "preprocess",
                    _preprocessor(numeric_features, categorical_features),
                ),
                ("model", classifier_model),
            ]
        ),
        Pipeline(
            [
                (
                    "preprocess",
                    _preprocessor(numeric_features, categorical_features),
                ),
                ("model", regressor_model),
            ]
        ),
    )


def _preprocessor(
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_features),
            ),
        ]
    )


def _metrics(
    labels: np.ndarray,
    realized: np.ndarray,
    probability: np.ndarray,
    expected: np.ndarray,
    *,
    baseline_probability: float,
    baseline_return: float,
) -> dict[str, Any]:
    auc = (
        float(roc_auc_score(labels, probability))
        if len(np.unique(labels)) == 2
        else 0.5
    )
    brier = float(brier_score_loss(labels, probability))
    baseline_brier = float(
        brier_score_loss(labels, np.full(len(labels), baseline_probability))
    )
    mae = float(mean_absolute_error(realized, expected))
    baseline_mae = float(
        mean_absolute_error(realized, np.full(len(realized), baseline_return))
    )
    correlation = (
        float(pd.Series(expected).corr(pd.Series(realized), method="spearman"))
        if len(realized) > 1
        else 0.0
    )
    correlation = correlation if math.isfinite(correlation) else 0.0
    cutoff = float(np.quantile(expected, 0.75))
    selected = realized[expected >= cutoff]
    all_mean = float(np.mean(realized))
    selected_mean = float(np.mean(selected)) if len(selected) else all_mean
    return {
        "roc_auc": auc,
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "delta_brier_vs_constant": baseline_brier - brier,
        "regression_mae": mae,
        "baseline_regression_mae": baseline_mae,
        "delta_mae_vs_constant": baseline_mae - mae,
        "regression_spearman": correlation,
        "all_candidate_mean_net_R": all_mean,
        "top_quartile_mean_net_R": selected_mean,
        "top_quartile_delta_net_R": selected_mean - all_mean,
        "top_quartile_count": len(selected),
    }


def _run_timeframe_ablations(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    family: str,
    random_state: int,
    baseline_probability: float,
    baseline_return: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for trial_index, (variant, timeframes) in enumerate(
        ABLATION_VARIANTS, start=1
    ):
        numeric, categorical = _ablation_features(timeframes)
        coverage = {
            timeframe: float(
                pd.to_numeric(
                    test[f"has_{timeframe}"], errors="coerce"
                ).fillna(0.0).mean()
            )
            for timeframe in timeframes
        }
        blockers = [
            f"{timeframe.upper()}_COVERAGE_BELOW_0_60"
            for timeframe, ratio in coverage.items()
            if ratio < 0.60
        ]
        if blockers:
            results.append(
                {
                    "trial_index": trial_index,
                    "variant": variant,
                    "timeframes": list(timeframes),
                    "features": [*numeric, *categorical],
                    "timeframe_coverage": coverage,
                    "status": "BLOCKED_INSUFFICIENT_CAUSAL_COVERAGE",
                    "blockers": blockers,
                    "trial_counted": True,
                    "model_fitted": False,
                    "financial_promotion_eligible": False,
                }
            )
            continue
        classifier, regressor = _pipelines(
            family,
            random_state=random_state,
            numeric_features=numeric,
            categorical_features=categorical,
        )
        features = [*numeric, *categorical]
        classifier.fit(training.loc[:, features], training["positive_net_trade"])
        regressor.fit(training.loc[:, features], training["native_exit_net_return"])
        probability = classifier.predict_proba(test.loc[:, features])[:, 1]
        expected = regressor.predict(test.loc[:, features])
        realized = test["native_exit_net_return"].to_numpy(dtype=float)
        metrics = _metrics(
            test["positive_net_trade"].to_numpy(dtype=int),
            realized,
            probability,
            expected,
            baseline_probability=baseline_probability,
            baseline_return=baseline_return,
        )
        diagnostics = _selection_diagnostics(test, expected)
        results.append(
            {
                "trial_index": trial_index,
                "variant": variant,
                "timeframes": list(timeframes),
                "features": features,
                "timeframe_coverage": coverage,
                "status": "OUTER_HOLDOUT_EVALUATED_RESEARCH_ONLY",
                "blockers": [
                    "FORWARD_EVIDENCE_GO_REQUIRED",
                    "EXTERNAL_DATA_GATES_REQUIRED",
                ],
                "trial_counted": True,
                "model_fitted": True,
                "financial_promotion_eligible": False,
                **metrics,
                **diagnostics,
            }
        )
    return results


def _blocked_timeframe_ablations(
    panel: pd.DataFrame, blockers: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "trial_index": index,
            "variant": variant,
            "timeframes": list(timeframes),
            "status": "BLOCKED_INSUFFICIENT_CANDIDATE_EVIDENCE",
            "blockers": blockers or ["CANDIDATE_PANEL_NOT_TRAINING_READY"],
            "trial_counted": True,
            "model_fitted": False,
            "financial_promotion_eligible": False,
            "panel_rows": len(panel),
        }
        for index, (variant, timeframes) in enumerate(ABLATION_VARIANTS, start=1)
    ]


def _run_interaction_trials(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    family: str,
    random_state: int,
    baseline_probability: float,
    baseline_return: float,
    timeframe_ablations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_numeric, base_categorical = _ablation_features(
        ("15m", "1h", "2h", "4h", "1d")
    )
    base = next(
        (
            row
            for row in timeframe_ablations
            if row["variant"] == "FULL_15M_1H_2H_4H_1D"
            and row.get("model_fitted") is True
        ),
        None,
    )
    results: list[dict[str, Any]] = []
    for trial_index, interaction in enumerate(
        INTERACTION_FEATURE_COLUMNS, start=1
    ):
        coverage = float(
            pd.to_numeric(test[interaction], errors="coerce").notna().mean()
        )
        blockers: list[str] = []
        if base is None:
            blockers.append("FULL_TIMEFRAME_BASELINE_NOT_EVALUABLE")
        if coverage < 0.60:
            blockers.append("INTERACTION_CAUSAL_COVERAGE_BELOW_0_60")
        if blockers:
            results.append(
                {
                    "trial_index": trial_index,
                    "interaction": interaction,
                    "status": "BLOCKED_INSUFFICIENT_CAUSAL_COVERAGE",
                    "coverage": coverage,
                    "blockers": blockers,
                    "trial_counted": True,
                    "model_fitted": False,
                    "financial_promotion_eligible": False,
                }
            )
            continue
        numeric = (*base_numeric, interaction)
        features = [*numeric, *base_categorical]
        classifier, regressor = _pipelines(
            family,
            random_state=random_state,
            numeric_features=numeric,
            categorical_features=base_categorical,
        )
        classifier.fit(training.loc[:, features], training["positive_net_trade"])
        regressor.fit(training.loc[:, features], training["native_exit_net_return"])
        probability = classifier.predict_proba(test.loc[:, features])[:, 1]
        expected = regressor.predict(test.loc[:, features])
        realized = test["native_exit_net_return"].to_numpy(dtype=float)
        metrics = _metrics(
            test["positive_net_trade"].to_numpy(dtype=int),
            realized,
            probability,
            expected,
            baseline_probability=baseline_probability,
            baseline_return=baseline_return,
        )
        results.append(
            {
                "trial_index": trial_index,
                "interaction": interaction,
                "status": "OUTER_HOLDOUT_EVALUATED_RESEARCH_ONLY",
                "coverage": coverage,
                "blockers": [
                    "MULTIPLICITY_CORRECTION_AND_NEW_FORWARD_CONFIRMATION_REQUIRED"
                ],
                "trial_counted": True,
                "model_fitted": True,
                "financial_promotion_eligible": False,
                "delta_top_quartile_net_R_vs_full_without_interaction": (
                    metrics["top_quartile_mean_net_R"]
                    - float(base["top_quartile_mean_net_R"])
                ),
                "delta_rank_ic_vs_full_without_interaction": (
                    metrics["regression_spearman"]
                    - float(base["regression_spearman"])
                ),
                **metrics,
                **_selection_diagnostics(test, expected),
            }
        )
    return results


def _blocked_interaction_trials(
    panel: pd.DataFrame, blockers: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "trial_index": index,
            "interaction": interaction,
            "status": "BLOCKED_INSUFFICIENT_CANDIDATE_EVIDENCE",
            "blockers": blockers or ["CANDIDATE_PANEL_NOT_TRAINING_READY"],
            "trial_counted": True,
            "model_fitted": False,
            "financial_promotion_eligible": False,
            "panel_rows": len(panel),
        }
        for index, interaction in enumerate(INTERACTION_FEATURE_COLUMNS, start=1)
    ]


def _ablation_features(
    timeframes: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = (
        *SCALAR_FEATURE_COLUMNS,
        *(
            name
            for name in TIMEFRAME_FEATURE_COLUMNS
            if not name.startswith("trend_state_")
            and any(name.endswith(f"_{timeframe}") for timeframe in timeframes)
        ),
    )
    categorical = (
        *CONTRACT_COLUMNS,
        *(
            name
            for name in TIMEFRAME_FEATURE_COLUMNS
            if name.startswith("trend_state_")
            and any(name.endswith(f"_{timeframe}") for timeframe in timeframes)
        ),
        "symbol",
        "strategy_id",
        "strategy_family",
        "strategy_dna_hash",
    )
    return numeric, categorical


def _selection_diagnostics(
    test: pd.DataFrame, expected: np.ndarray
) -> dict[str, Any]:
    work = test.loc[
        :,
        ["decision_timestamp", "symbol", "native_exit_net_return"],
    ].copy()
    work["score"] = expected
    work["decision_day"] = pd.to_datetime(
        work["decision_timestamp"], utc=True
    ).dt.floor("1D")
    selected = (
        work.sort_values(
            ["decision_day", "score"], ascending=[True, False]
        )
        .groupby("decision_day", as_index=False)
        .head(1)
        .sort_values("decision_day")
    )
    outcomes = selected["native_exit_net_return"].astype(float).to_numpy()
    equity = np.cumsum(outcomes)
    running_peak = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]
    drawdown = equity - running_peak
    symbols = selected["symbol"].astype(str).tolist()
    transitions = max(0, len(symbols) - 1)
    turnover = (
        sum(left != right for left, right in zip(symbols, symbols[1:], strict=False))
        / transitions
        if transitions
        else 0.0
    )
    rank_values = []
    for _, group in work.groupby("decision_timestamp"):
        if len(group) < 2:
            continue
        correlation = group["score"].corr(
            group["native_exit_net_return"], method="spearman"
        )
        if correlation is not None and math.isfinite(float(correlation)):
            rank_values.append(float(correlation))
    return {
        "outer_oos_rank_ic": float(np.mean(rank_values)) if rank_values else 0.0,
        "outer_oos_rank_ic_query_count": len(rank_values),
        "outer_oos_selected_net_R": float(outcomes.sum()) if len(outcomes) else 0.0,
        "outer_oos_selected_maximum_drawdown_R": (
            abs(float(drawdown.min())) if len(drawdown) else 0.0
        ),
        "selection_symbol_turnover_rate": float(turnover),
        "selection_turnover_semantics": (
            "TOP_CANDIDATE_SYMBOL_CHANGE_RATE_NOT_REALIZED_PORTFOLIO_TURNOVER"
        ),
        "selected_decision_day_count": len(selected),
    }


def _blocked_report(panel: pd.DataFrame, blockers: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "active_swing_candidate_model_tournament_v1",
        "status": "INSUFFICIENT_FORWARD_EVIDENCE",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
        "prediction_unit": "ONE_CANDIDATE_PER_PREDICTION",
        "panel_rows": len(panel),
        "training_blockers": blockers,
        "challengers": [],
        "model_fitted": False,
        "promotion_status": "NOT_ELIGIBLE",
        "performance_gate_go": False,
        "external_data_gate_go": False,
        "forward_evidence_go": False,
        "automatic_promotion": False,
        "financial_authority": False,
        "authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }
    report["content_hash"] = stable_hash(report)
    return report


__all__ = [
    "FEATURES",
    "MODEL_PATH",
    "OOS_PATH",
    "TOURNAMENT_PATH",
    "predict_active_swing_bundle",
    "run_active_swing_model_tournament",
]
