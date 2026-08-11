from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.quant_platform.ml import (
    ReturnPredictionModel,
    TemporalConvolutionalReturnModel,
)
from stocks.quant_platform.professional import ConstrainedQPortfolioPolicy
from stocks.quant_platform.regime import StatisticalRegimeDetector


EVIDENCE_PATH = Path("output/portfolio/learning-model-evidence.json")
MODEL_ROOT = Path("output/portfolio/models")
LEARNING_ROLES = {
    "supervised": {"capability_id": 17, "authority": "SHADOW_ONLY"},
    "tcn": {"capability_id": 17, "authority": "SHADOW_ONLY"},
    "unsupervised": {"capability_id": 10, "authority": "CONTEXT_ONLY"},
    "reinforcement_learning": {
        "capability_id": 32,
        "authority": "SHADOW_ONLY",
    },
}
MINIMUM_SHADOW_ROC_AUC = 0.52
MAXIMUM_SHADOW_BRIER_SCORE = 0.25


def train_shadow_learning_models(
    project_root: Path,
    *,
    symbols: Iterable[str] = ("AAPL", "NVDA"),
    sequence_length: int = 32,
    publish: bool = True,
) -> dict[str, Any]:
    """Train causal local shadow models and publish evidence, never authority."""

    root = project_root.resolve()
    symbol_reports: list[dict[str, Any]] = []
    serialized_models: dict[str, Any] = {}
    return_series: dict[str, pd.Series] = {}
    source_hashes: dict[str, str] = {}
    blockers: list[str] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper().strip()
        source = root / f"data/research/critical_trading/yfinance/{symbol}.parquet"
        if not source.is_file():
            blockers.append(f"PRICE_HISTORY_MISSING:{symbol}")
            continue
        try:
            frame = _price_frame(source)
            features, target, future_return = _features_and_target(frame)
            x, y, sequence_index = _sequences(
                features, target, length=sequence_length
            )
            supervised = ReturnPredictionModel(
                model_type="logistic", random_state=42, splits=4
            ).fit(features.loc[sequence_index], pd.Series(y, index=sequence_index))
            tcn = TemporalConvolutionalReturnModel(
                epochs=90, random_state=42, splits=3
            ).fit(x, pd.Series(y, index=sequence_index))
            regime_features = features.loc[
                :, ["momentum_20", "volatility_20", "drawdown_63", "volume_z_20"]
            ]
            regime_train = regime_features.iloc[: int(len(regime_features) * 0.8)]
            regime = StatisticalRegimeDetector(
                method="gmm", n_regimes=4, random_state=42
            ).fit(regime_train)
            latest_features = features.tail(1)
            latest_sequence = _latest_sequence(features, sequence_length)
            regime_prediction = regime.predict_proba(regime_features.tail(1)).iloc[0]
            supervised_probability = float(
                supervised.predict_proba(latest_features).iloc[0]
            )
            tcn_probability = float(tcn.predict_proba(latest_sequence)[0])
            supervised_report = supervised.report()
            tcn_report = tcn.report()
            supervised_gate = _classification_validation_gate(supervised_report)
            tcn_gate = _classification_validation_gate(tcn_report)
            latest_close = float(frame.iloc[-1]["close"])
            latest_expected_return = float(
                future_return.loc[sequence_index].tail(252).mean()
            )
            source_relative = source.relative_to(root).as_posix()
            source_hashes[source_relative] = sha256_file(source).upper()
            report = {
                "symbol": symbol,
                "data_source": source_relative,
                "trained_through": frame.index[-1].isoformat(),
                "training_observations": len(sequence_index),
                "sequence_length": sequence_length,
                "target": "FIVE_SESSION_RETURN_ABOVE_35_BPS_COST_STRESS",
                "latest_research_price_local": latest_close,
                "supervised": {
                    "probability_positive_after_costs": supervised_probability,
                    "validation_status": supervised_gate,
                    "report": supervised_report,
                },
                "tcn": {
                    "probability_positive_after_costs": tcn_probability,
                    "validation_status": tcn_gate,
                    "report": tcn_report,
                },
                "unsupervised": {
                    "regime": str(regime_prediction["regime"]),
                    "confidence": float(regime_prediction["confidence"]),
                    "method": "GAUSSIAN_MIXTURE_CAUSAL_TRAIN_WINDOW",
                    "interpretation": "RELATIVE_STATISTICAL_CLUSTER_NOT_OPERATIONAL_MACRO_REGIME",
                    "operational_regime_authority": False,
                },
                "historical_mean_forward_return_reference": latest_expected_return,
                "execution_authority": "NONE",
                "money_control": False,
            }
            report["prediction_hash"] = stable_hash(report)
            symbol_reports.append(report)
            serialized_models[symbol] = {
                "supervised": supervised,
                "tcn": tcn,
                "unsupervised": regime,
                "feature_names": list(features.columns),
                "sequence_length": sequence_length,
            }
            return_series[symbol] = frame["close"].pct_change()
        except (ValueError, KeyError, TypeError) as exc:
            blockers.append(f"LEARNING_TRAINING_BLOCKED:{symbol}:{type(exc).__name__}")

    rl_report: dict[str, Any] = {
        "status": "NOT_TRAINED",
        "execution_authority": "NONE",
        "money_control": False,
    }
    rl_weights: dict[str, float] = {}
    if len(return_series) >= 2:
        returns = pd.DataFrame(return_series).dropna().tail(1_500)
        signals = returns.rolling(20, min_periods=5).mean().fillna(0.0)
        try:
            policy = ConstrainedQPortfolioPolicy(
                episodes=45,
                maximum_asset_weight=0.25,
                transaction_cost_bps=7.0,
                random_state=42,
            ).fit(returns, signals=signals)
            state = {
                "signals": signals.iloc[-1].to_dict(),
                "volatility": returns.expanding().std(ddof=0).iloc[-1].to_dict(),
                "drawdown": 0.0,
            }
            _, rl_weights = policy.act(state)
            rl_report = {
                "status": (
                    "SHADOW_VALIDATION_GO"
                    if policy.validation.get("cumulative_net_return", 0) > 0
                    and policy.validation.get("mean_reward", 0) > 0
                    else "REJECTED_SHADOW_VALIDATION"
                ),
                "counterfactual_target_weights": rl_weights,
                "report": policy.report(),
                "execution_authority": "NONE",
                "money_control": False,
            }
            serialized_models["__portfolio_rl__"] = policy
        except (ValueError, KeyError, TypeError) as exc:
            blockers.append(f"RL_TRAINING_BLOCKED:{type(exc).__name__}")

    trained = bool(symbol_reports)
    accepted_classifiers = sum(
        prediction.get(model, {}).get("validation_status")
        == "SHADOW_VALIDATION_GO"
        for prediction in symbol_reports
        for model in ("supervised", "tcn")
    )
    accepted_rl = rl_report.get("status") == "SHADOW_VALIDATION_GO"
    body: dict[str, Any] = {
        "schema": "portfolio_learning_model_evidence_v1",
        "status": "GO" if trained else "NO_GO",
        "generated_at": _now(),
        "training_mode": "LOCAL_CAUSAL_SHADOW_RESEARCH",
        "source_hashes": source_hashes,
        "model_roles": LEARNING_ROLES,
        "symbol_predictions": symbol_reports,
        "portfolio_rl": rl_report,
        "rl_counterfactual_target_weights": rl_weights,
        "blockers": blockers,
        "classification_acceptance_gates": {
            "minimum_roc_auc": MINIMUM_SHADOW_ROC_AUC,
            "maximum_brier_score": MAXIMUM_SHADOW_BRIER_SCORE,
        },
        "accepted_shadow_classifier_count": accepted_classifiers,
        "rl_shadow_validation_go": accepted_rl,
        "financial_validation_status": (
            "SHADOW_VALIDATION_GO"
            if accepted_classifiers > 0 or accepted_rl
            else "NO_GO_NO_MODEL_CLEARS_TEMPORAL_ECONOMIC_GATES"
        ),
        "temporal_validation_required": True,
        "cost_aware_target": True,
        "model_predictions_may_grant_authority": False,
        "model_predictions_may_increase_quantity": False,
        "automatic_model_promotion": False,
        "research_only": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }
    body["evidence_hash"] = stable_hash(body)
    if publish and trained:
        model_path = root / MODEL_ROOT / f"{body['evidence_hash']}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(".joblib.tmp")
        joblib.dump(serialized_models, temporary)
        os.replace(temporary, model_path)
        body["model_artifact"] = model_path.relative_to(root).as_posix()
        body["model_artifact_hash"] = sha256_file(model_path).upper()
        body["evidence_hash"] = stable_hash(
            {key: value for key, value in body.items() if key != "evidence_hash"}
        )
        _atomic_json(root / EVIDENCE_PATH, body)
    return body


def integrate_learning_evidence(
    opportunities: Iterable[dict[str, Any]],
    evidence: dict[str, Any] | None,
    capability_authority: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach model counterfactuals without mutating financial opportunity fields."""

    raw = evidence or {}
    predictions = {
        str(row.get("symbol") or "").upper(): row
        for row in raw.get("symbol_predictions", [])
        if row.get("symbol")
    }
    capabilities = {
        int(row.get("id", 0)): str(row.get("authority") or row.get("role") or "")
        for row in capability_authority.get("capabilities", [])
    }
    contract_checks = {
        name: capabilities.get(int(spec["capability_id"])) == spec["authority"]
        for name, spec in LEARNING_ROLES.items()
    }
    contract_go = all(contract_checks.values())
    overlaid: list[dict[str, Any]] = []
    covered = 0
    for row in opportunities:
        symbol = str(row.get("symbol") or "").upper()
        prediction = predictions.get(symbol)
        if prediction:
            covered += 1
        overlay = {
            "schema": "learning_counterfactual_overlay_v1",
            "evidence_available": prediction is not None,
            "evidence_hash": raw.get("evidence_hash"),
            "financial_validation_status": raw.get(
                "financial_validation_status", "NO_GO_NOT_EVALUATED"
            ),
            "supervised_probability": (
                (prediction or {}).get("supervised", {}).get(
                    "probability_positive_after_costs"
                )
            ),
            "tcn_probability": (
                (prediction or {}).get("tcn", {}).get(
                    "probability_positive_after_costs"
                )
            ),
            "supervised_validation_status": (
                (prediction or {}).get("supervised", {}).get(
                    "validation_status"
                )
            ),
            "tcn_validation_status": (
                (prediction or {}).get("tcn", {}).get("validation_status")
            ),
            "unsupervised_regime": (
                (prediction or {}).get("unsupervised", {}).get("regime")
            ),
            "unsupervised_confidence": (
                (prediction or {}).get("unsupervised", {}).get("confidence")
            ),
            "rl_counterfactual_weight": raw.get(
                "rl_counterfactual_target_weights", {}
            ).get(symbol),
            "authority_contract_go": contract_go,
            "financial_fields_mutated": False,
            "may_grant_strategy_authority": False,
            "may_increase_quantity": False,
            "execution_influence": "NONE",
            "validated_for_execution_influence": False,
            "money_control": False,
        }
        overlaid.append({**row, "learning_overlay": overlay})
    report: dict[str, Any] = {
        "schema": "portfolio_learning_integration_v1",
        "status": "GO" if contract_go else "NO_GO",
        "evidence_status": raw.get("status", "NOT_TRAINED"),
        "evidence_hash": raw.get("evidence_hash"),
        "opportunity_count": len(overlaid),
        "covered_opportunity_count": covered,
        "model_roles": LEARNING_ROLES,
        "authority_contract_checks": contract_checks,
        "predictions_change_financial_fields": False,
        "predictions_change_strategy_authority": False,
        "predictions_change_execution_quantity": False,
        "execution_authority": "NONE",
        "broker_writes": 0,
    }
    report["content_hash"] = stable_hash(report)
    return overlaid, report


def load_learning_evidence(project_root: Path) -> dict[str, Any]:
    return _read_json(project_root / EVIDENCE_PATH)


def _price_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"session_date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError("price history schema incomplete")
    frame = frame.loc[:, sorted(required)].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"], utc=True)
    frame = frame.sort_values("session_date").drop_duplicates("session_date")
    frame = frame.set_index("session_date").apply(pd.to_numeric, errors="coerce")
    return frame.dropna().tail(1_800)


def _features_and_target(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    close = frame["close"]
    returns = close.pct_change()
    volume_log = np.log1p(frame["volume"])
    features = pd.DataFrame(
        {
            "return_1": returns,
            "momentum_5": close.pct_change(5),
            "momentum_20": close.pct_change(20),
            "volatility_20": returns.rolling(20).std(ddof=0),
            "intraday_range": (frame["high"] - frame["low"]) / close,
            "drawdown_63": close / close.rolling(63).max() - 1.0,
            "volume_z_20": (
                volume_log - volume_log.rolling(20).mean()
            )
            / volume_log.rolling(20).std(ddof=0),
        }
    ).replace([np.inf, -np.inf], np.nan)
    future_return = close.shift(-5) / close - 1.0
    target = (future_return > 0.0035).astype(float).where(future_return.notna())
    valid = features.dropna().index
    return features.loc[valid], target.reindex(valid), future_return.reindex(valid)


def _sequences(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    length: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    index: list[Any] = []
    values = features.to_numpy(dtype=float)
    for position in range(length - 1, len(features)):
        label = target.iloc[position]
        if pd.isna(label):
            continue
        rows.append(values[position - length + 1 : position + 1])
        labels.append(int(label))
        index.append(features.index[position])
    if not rows:
        raise ValueError("no causal learning sequences available")
    return np.asarray(rows), np.asarray(labels), pd.Index(index)


def _latest_sequence(features: pd.DataFrame, length: int) -> np.ndarray:
    if len(features) < length:
        raise ValueError("latest TCN sequence unavailable")
    return features.tail(length).to_numpy(dtype=float)[None, :, :]


def _classification_validation_gate(report: dict[str, Any]) -> str:
    validation = report.get("temporal_validation", {})
    roc_auc = float(validation.get("roc_auc", 0.0) or 0.0)
    brier = float(validation.get("brier_score", 1.0) or 1.0)
    return (
        "SHADOW_VALIDATION_GO"
        if roc_auc >= MINIMUM_SHADOW_ROC_AUC
        and brier <= MAXIMUM_SHADOW_BRIER_SCORE
        else "REJECTED_SHADOW_VALIDATION"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train causal shadow learning evidence")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "NVDA"])
    args = parser.parse_args()
    report = train_shadow_learning_models(args.root, symbols=args.symbols)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVIDENCE_PATH",
    "LEARNING_ROLES",
    "integrate_learning_evidence",
    "load_learning_evidence",
    "train_shadow_learning_models",
]
