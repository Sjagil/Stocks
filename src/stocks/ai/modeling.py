from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stocks.ai.panel import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from stocks.execution.idempotency import stable_hash


CLASSIFICATION_TARGET = "positive_net_trade"
REGRESSION_TARGET = "native_exit_net_return"
IDENTITY_COLUMNS = (
    "symbol",
    "security_id",
    "strategy_id",
    "fold_id",
    "decision_timestamp",
    "label_available_at",
)


@dataclass(frozen=True)
class TemporalFold:
    fold_id: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


@dataclass
class PlattCalibrator:
    model: LogisticRegression | None = None

    def fit(self, probability: np.ndarray, target: np.ndarray) -> PlattCalibrator:
        values = np.asarray(probability, dtype=float).clip(1e-6, 1 - 1e-6)
        labels = np.asarray(target, dtype=int)
        if len(np.unique(labels)) < 2:
            self.model = None
            return self
        logits = np.log(values / (1.0 - values)).reshape(-1, 1)
        self.model = LogisticRegression(C=1e6, max_iter=2_000).fit(logits, labels)
        return self

    def transform(self, probability: np.ndarray) -> np.ndarray:
        values = np.asarray(probability, dtype=float).clip(1e-6, 1 - 1e-6)
        if self.model is None:
            return values
        logits = np.log(values / (1.0 - values)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


def purged_walk_forward_splits(
    panel: pd.DataFrame,
    *,
    folds: int = 3,
    initial_fraction: float = 0.55,
    validation_fraction: float = 0.15,
    embargo_days: int = 5,
) -> Iterator[TemporalFold]:
    if folds < 2 or not 0.4 <= initial_fraction < 0.8:
        raise ValueError("invalid walk-forward configuration")
    decision = pd.to_datetime(panel["decision_timestamp"], utc=True)
    label_available = pd.to_datetime(panel["label_available_at"], utc=True)
    unique_dates = pd.DatetimeIndex(sorted(decision.dt.normalize().unique()))
    if len(unique_dates) < 100:
        raise ValueError("walk-forward requires at least 100 decision dates")
    starts = np.linspace(initial_fraction, 1.0, folds + 1)
    for number in range(folds):
        test_start_position = min(
            len(unique_dates) - 2, int(len(unique_dates) * starts[number])
        )
        test_end_position = (
            len(unique_dates)
            if number == folds - 1
            else int(len(unique_dates) * starts[number + 1])
        )
        test_start = unique_dates[test_start_position]
        test_end = unique_dates[test_end_position - 1] + pd.Timedelta(days=1)
        validation_length = max(30, int(test_start_position * validation_fraction))
        validation_start_position = max(1, test_start_position - validation_length)
        validation_start = unique_dates[validation_start_position]
        validation_end = test_start - pd.Timedelta(days=embargo_days)
        train_mask = (decision < validation_start) & (
            label_available < validation_start
        )
        validation_mask = (
            (decision >= validation_start)
            & (decision < validation_end)
            & (label_available < test_start)
        )
        test_mask = (decision >= test_start) & (decision < test_end)
        train_indices = np.flatnonzero(train_mask.to_numpy())
        validation_indices = np.flatnonzero(validation_mask.to_numpy())
        test_indices = np.flatnonzero(test_mask.to_numpy())
        if min(len(train_indices), len(validation_indices), len(test_indices)) < 40:
            continue
        yield TemporalFold(
            fold_id=f"OUTER_{number + 1:02d}",
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            train_start=decision.iloc[train_indices].min().isoformat(),
            train_end=decision.iloc[train_indices].max().isoformat(),
            validation_start=decision.iloc[validation_indices].min().isoformat(),
            validation_end=decision.iloc[validation_indices].max().isoformat(),
            test_start=decision.iloc[test_indices].min().isoformat(),
            test_end=decision.iloc[test_indices].max().isoformat(),
        )


def run_model_tournament(
    panel: pd.DataFrame,
    *,
    random_state: int = 42,
    external_data_gates: dict[str, bool] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    frame = _validated_panel(panel)
    classifier_templates = _classifier_templates(random_state)
    regressor_templates = _regressor_templates(random_state)
    oos_parts: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    selected_classifiers: list[str] = []
    selected_regressors: list[str] = []
    ranker_fold_reports: list[dict[str, Any]] = []
    for fold in purged_walk_forward_splits(frame):
        train = frame.iloc[fold.train_indices]
        validation = frame.iloc[fold.validation_indices]
        test = frame.iloc[fold.test_indices]
        x_train = train.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
        x_validation = validation.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
        x_test = test.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
        y_train_class = train[CLASSIFICATION_TARGET].astype(int)
        y_validation_class = validation[CLASSIFICATION_TARGET].astype(int)
        y_test_class = test[CLASSIFICATION_TARGET].astype(int)
        y_train_return = train[REGRESSION_TARGET].astype(float)
        y_validation_return = validation[REGRESSION_TARGET].astype(float)
        y_test_return = test[REGRESSION_TARGET].astype(float)

        classification_metrics: dict[str, Any] = {}
        calibrated_test: dict[str, np.ndarray] = {}
        validation_calibrated: dict[str, np.ndarray] = {}
        for name, template in classifier_templates.items():
            model = clone(template).fit(x_train, y_train_class)
            validation_raw = model.predict_proba(x_validation)[:, 1]
            calibrator = PlattCalibrator().fit(
                validation_raw, y_validation_class.to_numpy()
            )
            validation_probability = calibrator.transform(validation_raw)
            test_probability = calibrator.transform(
                model.predict_proba(x_test)[:, 1]
            )
            validation_calibrated[name] = validation_probability
            calibrated_test[name] = test_probability
            classification_metrics[name] = {
                "validation": _classification_metrics(
                    y_validation_class.to_numpy(), validation_probability
                ),
                "test": _classification_metrics(
                    y_test_class.to_numpy(), test_probability
                ),
            }
        selected_classifier = min(
            classification_metrics,
            key=lambda name: classification_metrics[name]["validation"][
                "brier_score"
            ],
        )
        selected_classifiers.append(selected_classifier)

        regression_metrics: dict[str, Any] = {}
        validation_returns: dict[str, np.ndarray] = {}
        test_returns: dict[str, np.ndarray] = {}
        for name, template in regressor_templates.items():
            model = clone(template).fit(x_train, y_train_return)
            validation_prediction = model.predict(x_validation)
            test_prediction = model.predict(x_test)
            validation_returns[name] = validation_prediction
            test_returns[name] = test_prediction
            regression_metrics[name] = {
                "validation": _regression_metrics(
                    y_validation_return.to_numpy(), validation_prediction
                ),
                "test": _regression_metrics(
                    y_test_return.to_numpy(), test_prediction
                ),
            }
        selected_regressor = max(
            regression_metrics,
            key=lambda name: regression_metrics[name]["validation"][
                "spearman_ic"
            ],
        )
        selected_regressors.append(selected_regressor)
        ranker_result = _fit_lightgbm_ranker(
            train,
            validation,
            test,
            features=[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
            random_state=random_state,
        )
        ranker_fold_reports.append(ranker_result["report"])
        validation_probability = validation_calibrated[selected_classifier]
        threshold = _select_meta_threshold(
            validation_probability, y_validation_return.to_numpy()
        )
        test_probability = calibrated_test[selected_classifier]
        test_expected_return = np.mean(
            np.column_stack(list(test_returns.values())), axis=1
        )
        validation_expected_return = np.mean(
            np.column_stack(list(validation_returns.values())), axis=1
        )
        conformal_radius = _conformal_radius(
            y_validation_return.to_numpy(), validation_expected_return
        )
        part = test.loc[:, [*IDENTITY_COLUMNS, CLASSIFICATION_TARGET, REGRESSION_TARGET]].copy()
        part["outer_fold"] = fold.fold_id
        for name, probability in calibrated_test.items():
            part[f"p_take_{name}"] = probability
        for name, prediction in test_returns.items():
            part[f"predicted_return_{name}"] = prediction
        part["selected_classifier"] = selected_classifier
        part["selected_regressor"] = selected_regressor
        part["p_take_calibrated"] = test_probability
        part["predicted_net_return"] = test_expected_return
        part["classifier_probability_std"] = np.std(
            np.column_stack(list(calibrated_test.values())), axis=1, ddof=0
        )
        part["regressor_prediction_std"] = np.std(
            np.column_stack(list(test_returns.values())), axis=1, ddof=0
        )
        part["return_interval_lower_90"] = (
            test_expected_return - conformal_radius
        )
        part["return_interval_upper_90"] = (
            test_expected_return + conformal_radius
        )
        part["return_interval_half_width_90"] = conformal_radius
        part["return_interval_contains_realized"] = (
            y_test_return.to_numpy() >= part["return_interval_lower_90"]
        ) & (y_test_return.to_numpy() <= part["return_interval_upper_90"])
        part["ranker_score"] = ranker_result["test_score"]
        part["meta_threshold"] = threshold
        part["meta_take"] = test_probability >= threshold
        part["model_abstained"] = (
            np.abs(test_probability - 0.5) < 0.08
        )
        oos_parts.append(part)
        fold_reports.append(
            {
                "fold": fold.__dict__,
                "selected_classifier": selected_classifier,
                "selected_regressor": selected_regressor,
                "meta_threshold": threshold,
                "conformal_return_radius_90": conformal_radius,
                "classification": classification_metrics,
                "regression": regression_metrics,
            }
        )

    if len(oos_parts) < 2:
        raise ValueError("insufficient valid purged outer folds")
    oos = pd.concat(oos_parts, ignore_index=True).sort_values(
        ["decision_timestamp", "strategy_id", "security_id"]
    )
    oos["ranking_score"] = oos["ranker_score"].where(
        oos["ranker_score"].notna(), oos["predicted_net_return"]
    )
    oos["cross_sectional_rank"] = oos.groupby(
        pd.to_datetime(oos["decision_timestamp"], utc=True)
    )["ranking_score"].rank(pct=True)
    oos["rank_take"] = oos["cross_sectional_rank"].ge(0.75)
    comparison = _same_period_comparison(oos, random_state=random_state)
    cost_stress = _cost_stress(oos)
    timeframe_ablations = timeframe_ablation_readiness(frame)
    uncertainty_evidence = _oos_uncertainty_evidence(oos)
    gates = external_data_gates or {}
    external_go = all(
        bool(gates.get(name))
        for name in ("PIT_DATA_GO", "SURVIVORSHIP_GO", "SHARIAH_PIT_GO")
    )
    performance_components = performance_gate_components(comparison)
    performance_components["OOS_RETURN_INTERVAL_COVERAGE_GE_088"] = (
        float(uncertainty_evidence["empirical_return_interval_coverage"])
        >= 0.88
    )
    performance_components["TIMEFRAME_ABLATION_ELIGIBLE_VARIANT"] = (
        int(timeframe_ablations["eligible_variant_count"]) > 0
    )
    active_swing_candidate_unit_go = active_swing_candidate_unit_readiness(frame)
    performance_components["ACTIVE_SWING_NATURAL_CANDIDATE_UNIT"] = (
        active_swing_candidate_unit_go
    )
    performance_go = all(performance_components.values())
    promotion_status = (
        "SHADOW_VALIDATION_GO"
        if performance_go and external_go
        else "SHADOW_ONLY_EXTERNAL_DATA_GATES_BLOCKED"
        if performance_go
        else "REJECTED_NO_INCREMENTAL_OOS_VALUE"
    )
    report: dict[str, Any] = {
        "schema": "global_decision_intelligence_tournament_v1",
        "status": "GO",
        "panel_rows": len(frame),
        "oos_rows": len(oos),
        "outer_fold_count": len(fold_reports),
        "folds": fold_reports,
        "classification_families": list(classifier_templates),
        "regression_families": list(regressor_templates),
        "ranking_family": "LIGHTGBM_RANKER",
        "ranking_query_semantics": "EXACT_DECISION_TIMESTAMP",
        "ranking_relevance_target": "WITHIN_QUERY_NET_RETURN_QUINTILE_0_TO_4",
        "ranker_folds": ranker_fold_reports,
        "selected_classifier_counts": pd.Series(selected_classifiers)
        .value_counts()
        .to_dict(),
        "selected_regressor_counts": pd.Series(selected_regressors)
        .value_counts()
        .to_dict(),
        "same_period_comparison": comparison,
        "cost_stress": cost_stress,
        "timeframe_ablations": timeframe_ablations,
        "uncertainty_evidence": uncertainty_evidence,
        "promotion_status": promotion_status,
        "performance_gate_go": performance_go,
        "performance_gate_components": performance_components,
        "external_data_gates": gates,
        "external_data_gate_go": external_go,
        "forward_evidence_go": False,
        "active_swing_candidate_unit_go": active_swing_candidate_unit_go,
        "candidate_unit_required": "ONE_NATURAL_STRATEGY_SETUP",
        "selection_conditioned_history": True,
        "research_only": True,
        "authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }
    report["content_hash"] = stable_hash(report)
    bundle = fit_final_shadow_bundle(
        frame,
        classifier_family=_mode(selected_classifiers),
        regressor_family=_mode(selected_regressors),
        random_state=random_state,
        report_hash=report["content_hash"],
    )
    return report, oos.reset_index(drop=True), bundle


def active_swing_candidate_unit_readiness(panel: pd.DataFrame) -> bool:
    required = {"candidate_unit", "candidate_identity"}
    if not required.issubset(panel.columns) or panel.empty:
        return False
    units = panel["candidate_unit"].astype(str)
    identities = panel["candidate_identity"].astype(str).str.strip()
    return bool(
        units.eq("ONE_NATURAL_STRATEGY_SETUP").all()
        and identities.ne("").all()
        and not identities.duplicated().any()
    )


def fit_final_shadow_bundle(
    panel: pd.DataFrame,
    *,
    classifier_family: str,
    regressor_family: str,
    random_state: int,
    report_hash: str,
) -> dict[str, Any]:
    frame = _validated_panel(panel)
    decision = pd.to_datetime(frame["decision_timestamp"], utc=True)
    dates = pd.DatetimeIndex(sorted(decision.dt.normalize().unique()))
    calibration_start = dates[int(len(dates) * 0.82)]
    label_available = pd.to_datetime(frame["label_available_at"], utc=True)
    train = frame.loc[(decision < calibration_start) & (label_available < calibration_start)]
    calibration = frame.loc[decision >= calibration_start]
    if min(len(train), len(calibration)) < 80:
        raise ValueError("final shadow model requires independent calibration data")
    features = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    classifier = clone(_classifier_templates(random_state)[classifier_family]).fit(
        train[features], train[CLASSIFICATION_TARGET].astype(int)
    )
    raw = classifier.predict_proba(calibration[features])[:, 1]
    calibrator = PlattCalibrator().fit(
        raw, calibration[CLASSIFICATION_TARGET].astype(int).to_numpy()
    )
    calibration_regressors = {
        name: clone(template).fit(
            train[features], train[REGRESSION_TARGET].astype(float)
        )
        for name, template in _regressor_templates(random_state).items()
    }
    calibration_prediction = np.mean(
        np.column_stack(
            [
                model.predict(calibration[features])
                for model in calibration_regressors.values()
            ]
        ),
        axis=1,
    )
    conformal_radius = _conformal_radius(
        calibration[REGRESSION_TARGET].to_numpy(dtype=float),
        calibration_prediction,
    )
    regressors = {
        name: clone(template).fit(frame[features], frame[REGRESSION_TARGET].astype(float))
        for name, template in _regressor_templates(random_state).items()
    }
    threshold = _select_meta_threshold(
        calibrator.transform(raw), calibration[REGRESSION_TARGET].to_numpy()
    )
    positive = frame.loc[frame[REGRESSION_TARGET] > 0, REGRESSION_TARGET]
    negative = frame.loc[frame[REGRESSION_TARGET] <= 0, REGRESSION_TARGET]
    return {
        "schema": "global_decision_intelligence_model_bundle_v1",
        "model_version": stable_hash(
            {
                "report": report_hash,
                "classifier": classifier_family,
                "regressor": regressor_family,
                "rows": len(frame),
            }
        )[:24],
        "classifier_family": classifier_family,
        "regressor_family": regressor_family,
        "classifier": classifier,
        "calibrator": calibrator,
        "regressors": regressors,
        "feature_columns": features,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "meta_threshold": threshold,
        "conformal_return_radius_90": conformal_radius,
        "uncertainty_method": (
            "CLASSIFICATION_ENTROPY_PLUS_MODEL_DISAGREEMENT_AND_90PCT_"
            "SPLIT_CONFORMAL_RETURN_INTERVAL"
        ),
        "expected_win": float(positive.mean()),
        "expected_loss": float(negative.mean()),
        "training_rows": len(frame),
        "training_symbols": sorted(frame["symbol"].astype(str).unique()),
        "calibration_start": calibration_start.isoformat(),
        "report_hash": report_hash,
        "authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
        "broker_writes": 0,
    }


def predict_model_bundle(
    bundle: dict[str, Any], features: pd.DataFrame
) -> pd.DataFrame:
    columns = list(bundle["feature_columns"])
    x = features.loc[:, columns]
    raw = bundle["classifier"].predict_proba(x)[:, 1]
    probability = bundle["calibrator"].transform(raw)
    regression_predictions = np.column_stack(
        [model.predict(x) for model in bundle["regressors"].values()]
    )
    expected = regression_predictions.mean(axis=1)
    disagreement = regression_predictions.std(axis=1, ddof=0)
    radius = float(bundle["conformal_return_radius_90"])
    entropy = -(
        probability * np.log(probability.clip(1e-12))
        + (1.0 - probability) * np.log((1.0 - probability).clip(1e-12))
    ) / math.log(2)
    normalized_disagreement = np.clip(
        disagreement / max(radius, 1e-12), 0.0, 1.0
    )
    return pd.DataFrame(
        {
            "p_take_raw": raw,
            "p_take_calibrated": probability,
            "predicted_net_return": expected,
            "uncertainty": np.maximum(entropy, normalized_disagreement),
            "model_disagreement": disagreement,
            "return_interval_lower_90": expected - radius,
            "return_interval_upper_90": expected + radius,
            "meta_take": probability >= float(bundle["meta_threshold"]),
            "model_abstained": np.abs(probability - 0.5) < 0.08,
        },
        index=features.index,
    )


def _classifier_templates(random_state: int) -> dict[str, Pipeline]:
    return {
        "LOGISTIC_BASELINE": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "EXTRA_TREES": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=240,
                        min_samples_leaf=8,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "HIST_GRADIENT_BOOSTING": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        min_samples_leaf=15,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "LIGHTGBM": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=240,
                        learning_rate=0.03,
                        num_leaves=15,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                        deterministic=True,
                        force_col_wise=True,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def _regressor_templates(random_state: int) -> dict[str, Pipeline]:
    return {
        "ELASTIC_NET_BASELINE": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.001,
                        l1_ratio=0.25,
                        max_iter=5_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "EXTRA_TREES": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=240,
                        min_samples_leaf=8,
                        max_features="sqrt",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "HIST_GRADIENT_BOOSTING": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=180,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        min_samples_leaf=15,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "LIGHTGBM": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    LGBMRegressor(
                        objective="regression_l1",
                        n_estimators=240,
                        learning_rate=0.03,
                        num_leaves=15,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=random_state,
                        n_jobs=1,
                        deterministic=True,
                        force_col_wise=True,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        sparse_threshold=0.0,
    )


def timeframe_ablation_readiness(panel: pd.DataFrame) -> dict[str, Any]:
    """Pre-register and gate ordered timeframe ablations on causal coverage.

    A blocked variant still counts as a research trial. It is not fitted on a
    tiny convenience subset because that would change the test population and
    create a misleading comparison with the deterministic baseline.
    """

    variants = {
        "15M_ONLY": ("return_15m",),
        "15M_PLUS_1H": ("return_15m", "return_1h"),
        "15M_PLUS_1H_PLUS_4H": ("return_15m", "return_1h", "return_4h"),
        "15M_PLUS_1H_PLUS_4H_PLUS_1D": (
            "return_15m",
            "return_1h",
            "return_4h",
            "return_1d",
        ),
        "FULL_15M_1H_2H_4H_1D": (
            "return_15m",
            "return_1h",
            "return_2h",
            "return_4h",
            "return_1d",
        ),
    }
    minimum_rows = 500
    minimum_decisions = 250
    minimum_coverage = 0.60
    rows: list[dict[str, Any]] = []
    for trial_index, (name, features) in enumerate(variants.items(), start=1):
        missing_columns = sorted(set(features) - set(panel.columns))
        complete = (
            pd.Series(False, index=panel.index)
            if missing_columns
            else panel.loc[:, list(features)].notna().all(axis=1)
        )
        complete_frame = panel.loc[complete]
        decision_count = (
            int(
                pd.to_datetime(
                    complete_frame["decision_timestamp"], utc=True
                ).nunique()
            )
            if "decision_timestamp" in complete_frame
            else 0
        )
        coverage = float(complete.mean()) if len(panel) else 0.0
        blockers = []
        if missing_columns:
            blockers.append("FEATURE_COLUMNS_MISSING")
        if int(complete.sum()) < minimum_rows:
            blockers.append("INSUFFICIENT_COMPLETE_ROWS")
        if decision_count < minimum_decisions:
            blockers.append("INSUFFICIENT_DECISION_TIMESTAMPS")
        if coverage < minimum_coverage:
            blockers.append("INSUFFICIENT_CAUSAL_COVERAGE")
        rows.append(
            {
                "trial_index": trial_index,
                "variant": name,
                "features": list(features),
                "trial_counted": True,
                "status": (
                    "ELIGIBLE_FOR_FIXED_OOS_ABLATION"
                    if not blockers
                    else "BLOCKED_INSUFFICIENT_CAUSAL_COVERAGE"
                ),
                "complete_rows": int(complete.sum()),
                "decision_timestamp_count": decision_count,
                "complete_case_coverage": round(coverage, 8),
                "blockers": blockers,
                "model_fitted": False,
                "financial_promotion_eligible": False,
            }
        )
    return {
        "schema": "timeframe_ablation_registry_v1",
        "comparison_population_rule": "SAME_FIXED_OOS_POPULATION_PER_VARIANT",
        "minimum_complete_rows": minimum_rows,
        "minimum_decision_timestamps": minimum_decisions,
        "minimum_complete_case_coverage": minimum_coverage,
        "variant_count": len(rows),
        "eligible_variant_count": sum(
            row["status"] == "ELIGIBLE_FOR_FIXED_OOS_ABLATION" for row in rows
        ),
        "blocked_variant_count": sum(
            row["status"] != "ELIGIBLE_FOR_FIXED_OOS_ABLATION" for row in rows
        ),
        "variants": rows,
        "authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
    }
def ranker_query_layout(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"decision_timestamp", REGRESSION_TARGET}
    if not required.issubset(frame.columns):
        raise ValueError(f"ranker query columns missing: {sorted(required - set(frame))}")
    work = frame.copy()
    work["decision_timestamp"] = pd.to_datetime(
        work["decision_timestamp"], utc=True, errors="raise"
    )
    work["_source_index"] = work.index
    order_columns = [
        "decision_timestamp",
        *[name for name in ("strategy_id", "security_id") if name in work],
    ]
    work = work.sort_values(order_columns, kind="stable").reset_index(drop=True)
    groups = work.groupby("decision_timestamp", sort=False, dropna=False)
    sizes = groups.size().astype(int).tolist()
    relevance = groups[REGRESSION_TARGET].transform(_query_relevance).astype(int)
    if sum(sizes) != len(work):
        raise ValueError("ranker group sizes do not cover all candidates")
    return {
        "ordered": work,
        "group_sizes": sizes,
        "relevance": relevance.to_numpy(dtype=int),
        "query_count": len(sizes),
        "multi_candidate_query_count": sum(size >= 2 for size in sizes),
        "singleton_query_count": sum(size == 1 for size in sizes),
        "maximum_query_size": max(sizes, default=0),
        "query_semantics": "EXACT_DECISION_TIMESTAMP",
    }


def _query_relevance(values: pd.Series) -> pd.Series:
    if len(values) == 1:
        return pd.Series(0, index=values.index, dtype=int)
    ranks = values.astype(float).rank(method="average", pct=True)
    return pd.Series(
        np.floor(
            np.clip(ranks.to_numpy() * 5.0 - 1e-12, 0, 4)
        ).astype(int),
        index=values.index,
    )


def _fit_lightgbm_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str],
    random_state: int,
) -> dict[str, Any]:
    train_layout = ranker_query_layout(train)
    validation_layout = ranker_query_layout(validation)
    test_layout = ranker_query_layout(test)
    if min(
        train_layout["multi_candidate_query_count"],
        validation_layout["multi_candidate_query_count"],
        test_layout["multi_candidate_query_count"],
    ) < 5:
        return {
            "test_score": np.full(len(test), np.nan),
            "report": {
                "status": "UNAVAILABLE_INSUFFICIENT_MULTI_CANDIDATE_QUERIES",
                "train": _query_diagnostics(train_layout),
                "validation": _query_diagnostics(validation_layout),
                "test": _query_diagnostics(test_layout),
            },
        }
    preprocessor = _preprocessor()
    ordered_train = train_layout["ordered"]
    ordered_validation = validation_layout["ordered"]
    ordered_test = test_layout["ordered"]
    x_train = preprocessor.fit_transform(ordered_train[features])
    x_validation = preprocessor.transform(ordered_validation[features])
    x_test = preprocessor.transform(ordered_test[features])
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=(0, 1, 3, 7, 15),
        n_estimators=240,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=15,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    ranker.fit(
        x_train,
        train_layout["relevance"],
        group=train_layout["group_sizes"],
        eval_X=x_validation,
        eval_y=validation_layout["relevance"],
        eval_group=[validation_layout["group_sizes"]],
        eval_at=(1, 3, 5),
    )
    validation_score = ranker.predict(x_validation)
    test_score_ordered = ranker.predict(x_test)
    test_score = pd.Series(
        test_score_ordered,
        index=ordered_test["_source_index"].to_numpy(),
    ).reindex(test.index).to_numpy(dtype=float)
    return {
        "test_score": test_score,
        "report": {
            "status": "GO",
            "train": _query_diagnostics(train_layout),
            "validation": {
                **_query_diagnostics(validation_layout),
                **_ranking_metrics(ordered_validation, validation_score),
            },
            "test": {
                **_query_diagnostics(test_layout),
                **_ranking_metrics(ordered_test, test_score_ordered),
            },
            "execution_authority": "NONE",
        },
    }


def _query_diagnostics(layout: dict[str, Any]) -> dict[str, Any]:
    return {
        key: layout[key]
        for key in (
            "query_count",
            "multi_candidate_query_count",
            "singleton_query_count",
            "maximum_query_size",
            "query_semantics",
        )
    }


def _ranking_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    work = frame[["decision_timestamp", REGRESSION_TARGET]].copy()
    work["score"] = np.asarray(score, dtype=float)
    values = [
        _safe_spearman(group["score"].to_numpy(), group[REGRESSION_TARGET].to_numpy())
        for _, group in work.groupby("decision_timestamp", sort=False)
        if len(group) >= 2
    ]
    return {
        "mean_rank_ic": float(np.mean(values)) if values else 0.0,
        "rank_ic_query_count": len(values),
    }


def _validated_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        *IDENTITY_COLUMNS,
        CLASSIFICATION_TARGET,
        REGRESSION_TARGET,
    }
    if not required.issubset(panel.columns):
        raise ValueError(f"ML panel missing columns: {sorted(required - set(panel))}")
    frame = panel.copy()
    frame["decision_timestamp"] = pd.to_datetime(
        frame["decision_timestamp"], utc=True, errors="coerce"
    )
    frame["label_available_at"] = pd.to_datetime(
        frame["label_available_at"], utc=True, errors="coerce"
    )
    frame = frame.dropna(
        subset=["decision_timestamp", "label_available_at", REGRESSION_TARGET]
    ).sort_values("decision_timestamp")
    if len(frame) < 500 or frame[CLASSIFICATION_TARGET].nunique() != 2:
        raise ValueError("global ML panel has insufficient class-balanced evidence")
    return frame.reset_index(drop=True)


def _classification_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    values = np.asarray(probability, dtype=float).clip(1e-8, 1 - 1e-8)
    labels = np.asarray(target, dtype=int)
    auc = float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else 0.5
    return {
        "brier_score": float(brier_score_loss(labels, values)),
        "log_loss": float(log_loss(labels, values, labels=[0, 1])),
        "roc_auc": auc,
        "ece_10": _expected_calibration_error(labels, values, bins=10),
        "observations": float(len(labels)),
    }


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlation = _safe_spearman(target, prediction)
    return {
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "spearman_ic": correlation,
        "observations": float(len(target)),
    }


def _expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, *, bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper
        )
        if mask.any():
            result += mask.mean() * abs(target[mask].mean() - probability[mask].mean())
    return float(result)


def _select_meta_threshold(probability: np.ndarray, returns: np.ndarray) -> float:
    best_threshold = 0.55
    best_score = -math.inf
    for threshold in np.linspace(0.40, 0.70, 13):
        selected = probability >= threshold
        if selected.mean() < 0.20 or selected.sum() < 25:
            continue
        selected_mean = float(np.mean(returns[selected]))
        score = selected_mean - 0.10 * float(np.std(returns[selected], ddof=0))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _same_period_comparison(oos: pd.DataFrame, *, random_state: int) -> dict[str, Any]:
    returns = oos[REGRESSION_TARGET].to_numpy(dtype=float)
    meta_mask = oos["meta_take"].to_numpy(dtype=bool)
    rank_mask = oos["rank_take"].fillna(False).to_numpy(dtype=bool)
    baseline = _decision_metrics(returns, np.ones(len(oos), dtype=bool))
    meta = _decision_metrics(returns, meta_mask)
    ranking = _decision_metrics(returns, rank_mask)
    meta_effect = np.where(meta_mask, returns, 0.0) - returns
    ranking_effect = np.where(rank_mask, returns, 0.0) - returns
    rank_ics: list[float] = []
    queries = pd.to_datetime(oos["decision_timestamp"], utc=True)
    for _, group in oos.groupby(queries):
        if len(group) < 3:
            continue
        value = _safe_spearman(
            group["ranking_score"].to_numpy(),
            group[REGRESSION_TARGET].to_numpy(),
        )
        rank_ics.append(value)
    return {
        "deterministic": baseline,
        "meta_label": {
            **meta,
            "delta_net_expectancy": meta["net_expectancy"] - baseline["net_expectancy"],
            "delta_trade_count": meta["trade_count"] - baseline["trade_count"],
            "bootstrap_probability_of_improvement": _bootstrap_probability(
                meta_effect, random_state=random_state
            ),
        },
        "ranking": {
            **ranking,
            "delta_net_expectancy": ranking["net_expectancy"] - baseline["net_expectancy"],
            "bootstrap_probability_of_improvement": _bootstrap_probability(
                ranking_effect, random_state=random_state + 1
            ),
            "mean_rank_ic": float(np.mean(rank_ics)) if rank_ics else 0.0,
            "rank_ic_period_count": len(rank_ics),
        },
    }


def performance_gate_components(
    comparison: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    meta = comparison["meta_label"]
    ranking = comparison["ranking"]
    return {
        "META_DELTA_NET_EXPECTANCY_POSITIVE": (
            float(meta["delta_net_expectancy"]) > 0
        ),
        "META_BOOTSTRAP_PROBABILITY_GE_095": (
            float(meta["bootstrap_probability_of_improvement"]) >= 0.95
        ),
        "RANK_IC_POSITIVE": float(ranking["mean_rank_ic"]) > 0,
        "RANKING_DELTA_NET_EXPECTANCY_POSITIVE": (
            float(ranking["delta_net_expectancy"]) > 0
        ),
        "RANKING_BOOTSTRAP_PROBABILITY_GE_095": (
            float(ranking["bootstrap_probability_of_improvement"]) >= 0.95
        ),
    }


def _decision_metrics(returns: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    realized = np.where(mask, returns, 0.0)
    selected = returns[mask]
    equity = np.cumprod(1.0 + realized)
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0
    gains = selected[selected > 0].sum()
    losses = -selected[selected < 0].sum()
    return {
        "trade_count": int(mask.sum()),
        "coverage": float(mask.mean()),
        "net_expectancy": float(selected.mean()) if len(selected) else 0.0,
        "aggregate_candidate_return": float(realized.sum()),
        "compounded_candidate_path_return": float(equity[-1] - 1.0),
        "maximum_drawdown": float(drawdown.min()),
        "hit_rate": float((selected > 0).mean()) if len(selected) else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else None,
    }


def _conformal_radius(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.abs(
        np.asarray(target, dtype=float) - np.asarray(prediction, dtype=float)
    )
    residual = residual[np.isfinite(residual)]
    if len(residual) < 25:
        raise ValueError("conformal interval requires at least 25 residuals")
    return float(np.quantile(residual, 0.90, method="higher"))


def _oos_uncertainty_evidence(oos: pd.DataFrame) -> dict[str, Any]:
    required = {
        "classifier_probability_std",
        "regressor_prediction_std",
        "return_interval_lower_90",
        "return_interval_upper_90",
        "return_interval_contains_realized",
    }
    missing = sorted(required - set(oos.columns))
    if missing:
        raise ValueError(f"OOF uncertainty columns missing: {missing}")
    width = (
        pd.to_numeric(oos["return_interval_upper_90"], errors="coerce")
        - pd.to_numeric(oos["return_interval_lower_90"], errors="coerce")
    )
    empirical_coverage = float(
        oos["return_interval_contains_realized"].astype(bool).mean()
    )
    return {
        "schema": "oos_uncertainty_evidence_v1",
        "method": "VALIDATION_ONLY_90PCT_SPLIT_CONFORMAL_PER_OUTER_FOLD",
        "oos_rows": len(oos),
        "nominal_return_interval_coverage": 0.90,
        "minimum_promotion_coverage": 0.88,
        "empirical_return_interval_coverage": empirical_coverage,
        "coverage_gap_to_nominal": empirical_coverage - 0.90,
        "status": (
            "CALIBRATION_COVERAGE_GO"
            if empirical_coverage >= 0.88
            else "UNDER_COVERAGE_RESEARCH_ONLY"
        ),
        "mean_return_interval_width": float(width.mean()),
        "median_return_interval_width": float(width.median()),
        "mean_classifier_probability_std": float(
            pd.to_numeric(
                oos["classifier_probability_std"], errors="coerce"
            ).mean()
        ),
        "mean_regressor_prediction_std": float(
            pd.to_numeric(
                oos["regressor_prediction_std"], errors="coerce"
            ).mean()
        ),
        "future_test_labels_used_for_interval_fit": False,
        "financial_promotion_authority": False,
    }


def _bootstrap_probability(values: np.ndarray, *, random_state: int) -> float:
    rng = np.random.default_rng(random_state)
    if len(values) < 30:
        return 0.0
    means = np.empty(1_000)
    for index in range(len(means)):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float((means > 0.0).mean())


def _cost_stress(oos: pd.DataFrame) -> dict[str, Any]:
    base_returns = oos[REGRESSION_TARGET].to_numpy(dtype=float)
    estimated_cost = 0.002
    result: dict[str, Any] = {}
    for multiplier in (1.0, 1.5, 2.0):
        stressed = base_returns - estimated_cost * (multiplier - 1.0)
        result[f"{multiplier:.1f}x"] = {
            "meta_label": _decision_metrics(
                stressed, oos["meta_take"].to_numpy(dtype=bool)
            ),
            "ranking": _decision_metrics(
                stressed, oos["rank_take"].fillna(False).to_numpy(dtype=bool)
            ),
        }
    return result


def _mode(values: list[str]) -> str:
    if not values:
        raise ValueError("model selection is empty")
    return str(pd.Series(values).value_counts().sort_values(ascending=False).index[0])


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if finite.sum() < 3:
        return 0.0
    left_values = left_values[finite]
    right_values = right_values[finite]
    if np.ptp(left_values) == 0.0 or np.ptp(right_values) == 0.0:
        return 0.0
    value = spearmanr(left_values, right_values).statistic
    return float(value) if np.isfinite(value) else 0.0


__all__ = [
    "CLASSIFICATION_TARGET",
    "PlattCalibrator",
    "REGRESSION_TARGET",
    "TemporalFold",
    "fit_final_shadow_bundle",
    "predict_model_bundle",
    "performance_gate_components",
    "purged_walk_forward_splits",
    "ranker_query_layout",
    "run_model_tournament",
]
