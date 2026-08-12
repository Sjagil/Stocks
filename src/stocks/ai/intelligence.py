from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

from stocks.ai.contracts import ModelEvidence
from stocks.ai.active_swing_panel import (
    build_active_swing_candidate_panel,
    infer_current_active_swing_candidates,
)
from stocks.ai.active_swing_modeling import (
    MODEL_PATH as ACTIVE_SWING_MODEL_PATH,
    OOS_PATH as ACTIVE_SWING_OOS_PATH,
    TOURNAMENT_PATH as ACTIVE_SWING_TOURNAMENT_PATH,
    run_active_swing_model_tournament,
)
from stocks.ai.modeling import predict_model_bundle, run_model_tournament
from stocks.ai.panel import (
    CATEGORICAL_FEATURES,
    DAILY_NUMERIC_FEATURES,
    MULTITIMEFRAME_RETURN_FEATURES,
    NUMERIC_FEATURES,
    PRICE_ROOT,
    SECURITY_MASTER_PATH,
    build_canonical_ml_panel,
    build_causal_bar_features,
)
from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.p3.io import atomic_write_json, read_json
from stocks.rl.data import (
    build_causal_multitimeframe_context,
    load_multitimeframe_frames,
)


OUTPUT_ROOT = Path("output/ai/decision-intelligence")
TOURNAMENT_PATH = OUTPUT_ROOT / "tournament.json"
OOS_PATH = OUTPUT_ROOT / "oos-predictions.parquet"
MODEL_PATH = OUTPUT_ROOT / "shadow-model.joblib"
MODEL_MANIFEST_PATH = OUTPUT_ROOT / "model-manifest.json"
CURRENT_INFERENCE_PATH = OUTPUT_ROOT / "current-inference.json"
COUNTERFACTUAL_PATH = OUTPUT_ROOT / "counterfactual-status.json"
REFRESH_STATUS_PATH = OUTPUT_ROOT / "refresh-status.json"
OPPORTUNITIES_PATH = Path("output/portfolio/normalized-opportunities.json")


def enqueue_refresh_if_due(
    project_root: Path,
    *,
    maximum_age_hours: float = 24.0,
) -> dict[str, Any]:
    """Dispatch a stale refresh in a detached process and return immediately."""

    root = project_root.resolve()
    manifest = read_json(root / MODEL_MANIFEST_PATH)
    age_hours = _manifest_age_hours(manifest)
    if age_hours < maximum_age_hours:
        return {
            "schema": "decision_intelligence_refresh_dispatch_v1",
            "status": "SKIPPED_FRESH",
            "age_hours": age_hours,
            "model_version": manifest.get("model_version"),
            "promotion_status": manifest.get("promotion_status"),
            "execution_authority": "NONE",
            "broker_writes": 0,
        }
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0
    )
    process = subprocess.Popen(
        [sys.executable, str(root / "main.py"), "ai", "refresh-if-due"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=flags,
        start_new_session=not bool(flags),
    )
    status = {
        "schema": "decision_intelligence_refresh_dispatch_v1",
        "status": "ENQUEUED",
        "process_id": process.pid,
        "enqueued_at": datetime.now(UTC).isoformat(),
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }
    atomic_write_json(root / OUTPUT_ROOT / "refresh-dispatch.json", status)
    return status


def refresh_if_due(
    project_root: Path,
    *,
    maximum_age_hours: float = 24.0,
) -> dict[str, Any]:
    """Run an independently locked refresh only when its manifest is stale."""

    root = project_root.resolve()
    manifest = read_json(root / MODEL_MANIFEST_PATH)
    age_hours = _manifest_age_hours(manifest)
    if age_hours < maximum_age_hours:
        return {
            "schema": "decision_intelligence_refresh_status_v1",
            "status": "SKIPPED_FRESH",
            "age_hours": age_hours,
            "model_version": manifest.get("model_version"),
            "promotion_status": manifest.get("promotion_status"),
            "execution_authority": "NONE",
            "broker_writes": 0,
        }
    lock = FileLock(str(root / OUTPUT_ROOT / "refresh.lock"), timeout=0)
    try:
        with lock:
            started = datetime.now(UTC)
            atomic_write_json(
                root / REFRESH_STATUS_PATH,
                {
                    "schema": "decision_intelligence_refresh_status_v1",
                    "status": "RUNNING",
                    "started_at": started.isoformat(),
                    "execution_authority": "NONE",
                    "broker_writes": 0,
                },
            )
            result = refresh_decision_intelligence(root, publish=True)
            status = {
                "schema": "decision_intelligence_refresh_status_v1",
                "status": "COMPLETED",
                "started_at": started.isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
                "model_version": result["manifest"]["model_version"],
                "promotion_status": result["tournament"]["promotion_status"],
                "evidence_count": result["current_inference"]["evidence_count"],
                "execution_authority": "NONE",
                "broker_calls": 0,
                "broker_writes": 0,
            }
            atomic_write_json(root / REFRESH_STATUS_PATH, status)
            return status
    except Timeout:
        return {
            "schema": "decision_intelligence_refresh_status_v1",
            "status": "SKIPPED_BUSY",
            "execution_authority": "NONE",
            "broker_writes": 0,
        }
    except Exception as exc:
        failed = {
            "schema": "decision_intelligence_refresh_status_v1",
            "status": "FAILED",
            "failed_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }
        atomic_write_json(root / REFRESH_STATUS_PATH, failed)
        raise


def refresh_decision_intelligence(
    project_root: Path,
    *,
    publish: bool = True,
) -> dict[str, Any]:
    """Retrain and infer one global shadow model without broker or money authority."""

    root = project_root.resolve()
    active_swing_panel, active_swing_panel_status = (
        build_active_swing_candidate_panel(root, publish=publish)
    )
    active_swing_tournament, active_swing_oos, active_swing_bundle = (
        run_active_swing_model_tournament(
            active_swing_panel,
            active_swing_panel_status,
        )
    )
    panel, panel_status = build_canonical_ml_panel(root, publish=publish)
    data_gates = {
        "PIT_DATA_GO": bool(panel_status.get("point_in_time_universe_complete")),
        "SURVIVORSHIP_GO": False,
        "SHARIAH_PIT_GO": bool(panel_status.get("shariah_history_complete")),
    }
    tournament, oos, bundle = run_model_tournament(
        panel, external_data_gates=data_gates
    )
    inference = infer_current_opportunities(root, bundle, tournament)
    active_swing_inference = infer_current_active_swing_candidates(
        root,
        active_swing_bundle or {},
        active_swing_tournament,
        publish=publish,
    )
    counterfactual = _counterfactual_status(tournament, inference)
    manifest: dict[str, Any] = {
        "schema": "global_decision_intelligence_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": bundle["model_version"],
        "classifier_family": bundle["classifier_family"],
        "regressor_family": bundle["regressor_family"],
        "training_rows": bundle["training_rows"],
        "panel_hash": panel_status.get("panel_sha256"),
        "active_swing_panel_hash": active_swing_panel_status.get("panel_sha256"),
        "active_swing_training_rows": len(active_swing_panel),
        "active_swing_training_ready": active_swing_panel_status.get(
            "training_ready", False
        ),
        "active_swing_model_status": active_swing_tournament.get("status"),
        "active_swing_model_version": active_swing_tournament.get("model_version"),
        "active_swing_model_artifact_current": active_swing_bundle is not None,
        "active_swing_current_evidence_count": active_swing_inference.get(
            "evidence_count", 0
        ),
        "tournament_hash": tournament["content_hash"],
        "status": tournament["promotion_status"],
        "promotion_status": tournament["promotion_status"],
        "current_evidence_count": len(inference["model_evidence"]),
        "authority": "SHADOW_ONLY",
        "automatic_promotion": False,
        "money_control": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }
    if publish:
        atomic_write_json(root / ACTIVE_SWING_TOURNAMENT_PATH, active_swing_tournament)
        if active_swing_bundle is not None:
            _atomic_parquet(root / ACTIVE_SWING_OOS_PATH, active_swing_oos)
            _atomic_joblib(root / ACTIVE_SWING_MODEL_PATH, active_swing_bundle)
            manifest["active_swing_oos_sha256"] = sha256_file(
                root / ACTIVE_SWING_OOS_PATH
            ).upper()
            manifest["active_swing_model_sha256"] = sha256_file(
                root / ACTIVE_SWING_MODEL_PATH
            ).upper()
        _atomic_parquet(root / OOS_PATH, oos)
        _atomic_joblib(root / MODEL_PATH, bundle)
        manifest["oos_sha256"] = sha256_file(root / OOS_PATH).upper()
        manifest["model_sha256"] = sha256_file(root / MODEL_PATH).upper()
        manifest["content_hash"] = stable_hash(manifest)
        atomic_write_json(root / TOURNAMENT_PATH, tournament)
        atomic_write_json(root / CURRENT_INFERENCE_PATH, inference)
        atomic_write_json(root / COUNTERFACTUAL_PATH, counterfactual)
        atomic_write_json(root / MODEL_MANIFEST_PATH, manifest)
    return {
        "panel": panel_status,
        "active_swing_panel": active_swing_panel_status,
        "active_swing_tournament": active_swing_tournament,
        "tournament": tournament,
        "manifest": manifest,
        "current_inference": inference,
        "current_active_swing_inference": active_swing_inference,
        "counterfactual": counterfactual,
    }


def infer_current_opportunities(
    project_root: Path,
    bundle: dict[str, Any],
    tournament: dict[str, Any],
) -> dict[str, Any]:
    root = project_root.resolve()
    payload = read_json(root / OPPORTUNITIES_PATH)
    opportunities = payload.get("combined_ranking", [])
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in opportunities:
        symbol = str(row.get("symbol") or "").upper().strip()
        if symbol and symbol != "CASH" and symbol not in by_symbol:
            by_symbol[symbol] = row
    feature_frame, missing = _latest_opportunity_features(root, by_symbol)
    if feature_frame.empty:
        evidence: list[dict[str, Any]] = []
    else:
        predictions = predict_model_bundle(bundle, feature_frame)
        predictions["cross_sectional_rank"] = predictions[
            "predicted_net_return"
        ].rank(pct=True)
        evidence = []
        for index, features in feature_frame.iterrows():
            prediction = predictions.loc[index]
            probability = float(prediction["p_take_calibrated"])
            conservative_ev = (
                probability * float(bundle["expected_win"])
                + (1.0 - probability) * float(bundle["expected_loss"])
            )
            feature_hash = stable_hash(
                {
                    name: _json_scalar(features[name])
                    for name in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
                }
            )
            out_of_distribution = bool(
                features["symbol"] not in set(bundle.get("training_symbols", ()))
                or float(features["missing_feature_fraction"]) > 0.20
            )
            record = ModelEvidence(
                evidence_id=stable_hash(
                    {
                        "model": bundle["model_version"],
                        "symbol": features["symbol"],
                        "feature": feature_hash,
                    }
                )[:32],
                model_version=bundle["model_version"],
                symbol=str(features["symbol"]),
                as_of=pd.Timestamp(features["as_of"]).to_pydatetime(),
                feature_timestamp=pd.Timestamp(
                    features["feature_timestamp"]
                ).to_pydatetime(),
                probability_positive_net=probability,
                predicted_net_return=float(prediction["predicted_net_return"]),
                expected_win=float(bundle["expected_win"]),
                expected_loss=float(bundle["expected_loss"]),
                conservative_expected_value=conservative_ev,
                uncertainty=float(prediction["uncertainty"]),
                model_disagreement=float(prediction["model_disagreement"]),
                return_interval_lower_90=float(
                    prediction["return_interval_lower_90"]
                ),
                return_interval_upper_90=float(
                    prediction["return_interval_upper_90"]
                ),
                cross_sectional_rank=float(prediction["cross_sectional_rank"]),
                meta_take=bool(prediction["meta_take"]),
                abstained=bool(prediction["model_abstained"]),
                out_of_distribution=out_of_distribution,
                validation_status=str(tournament["promotion_status"]),
                tournament_hash=str(tournament["content_hash"]),
                feature_hash=feature_hash,
            )
            evidence.append(record.model_dump(mode="json"))
    return {
        "schema": "current_global_decision_intelligence_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": bundle["model_version"],
        "tournament_hash": tournament["content_hash"],
        "promotion_status": tournament["promotion_status"],
        "opportunity_count": len(by_symbol),
        "evidence_count": len(evidence),
        "missing_price_history": sorted(missing),
        "model_evidence": evidence,
        "consumer_contract": "TYPED_NON_FINANCIAL_OVERLAY",
        "financial_fields_mutated": False,
        "automatic_promotion": False,
        "authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }


def _latest_opportunity_features(
    root: Path, opportunities: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, list[str]]:
    identities: dict[str, dict[str, Any]] = {}
    master_path = root / SECURITY_MASTER_PATH
    if master_path.is_file():
        master = pd.read_parquet(master_path)
        identities = {
            str(row["ticker"]).upper(): row.to_dict()
            for _, row in master.iterrows()
        }
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for symbol in opportunities:
        path = root / PRICE_ROOT / f"{symbol}.parquet"
        if not path.is_file():
            missing.append(symbol)
            continue
        frame = pd.read_parquet(path).rename(columns={"session_date": "date"})
        if not {"date", "open", "high", "low", "close", "volume"}.issubset(frame):
            missing.append(symbol)
            continue
        identity = identities.get(symbol, {})
        frame["security_id"] = str(identity.get("security_id") or f"SYMBOL:{symbol}")
        frame["ticker"] = symbol
        frame["sector"] = str(identity.get("sector") or "UNKNOWN")
        frame["currency"] = str(identity.get("currency") or "USD")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), missing
    featured = build_causal_bar_features(pd.concat(frames, ignore_index=True))
    rows: list[dict[str, Any]] = []
    for symbol, opportunity in opportunities.items():
        history = featured.loc[featured["ticker"].eq(symbol)].sort_values("date")
        if history.empty:
            continue
        context = history.iloc[-1]
        observed = [
            name
            for name in DAILY_NUMERIC_FEATURES
            if name not in {"estimated_round_trip_cost_rate", "missing_feature_fraction"}
        ]
        numeric = {name: _finite_or_nan(context.get(name)) for name in observed}
        numeric["estimated_round_trip_cost_rate"] = max(
            0.002,
            2.0 * float(opportunity.get("estimated_slippage_bps") or 0.0) / 10_000.0,
        )
        feature_time = pd.Timestamp(context["date"])
        if feature_time.tzinfo is None:
            feature_time = feature_time.tz_localize("UTC")
        else:
            feature_time = feature_time.tz_convert("UTC")
        signal_time = pd.to_datetime(
            opportunity.get("signal_timestamp"), utc=True, errors="coerce"
        )
        as_of = signal_time if not pd.isna(signal_time) else pd.Timestamp.now(tz="UTC")
        as_of = max(as_of, feature_time)
        mtf = build_causal_multitimeframe_context(
            pd.Series([as_of]),
            load_multitimeframe_frames(root, symbol),
        ).iloc[0]
        for name in MULTITIMEFRAME_RETURN_FEATURES:
            numeric[name] = _finite_or_nan(mtf.get(name))
            numeric[f"missing__{name}"] = int(pd.isna(numeric[name]))
        all_observed = [*observed, *MULTITIMEFRAME_RETURN_FEATURES]
        numeric["missing_feature_fraction"] = float(
            np.mean([pd.isna(numeric[name]) for name in all_observed])
        )
        rows.append(
            {
                "symbol": symbol,
                "strategy_id": str(opportunity.get("source") or "CURRENT_OPPORTUNITY"),
                "strategy_family": str(opportunity.get("strategy_family") or "UNKNOWN"),
                "entry_timeframe": str(opportunity.get("timeframe") or "UNKNOWN"),
                "asset_class": str(opportunity.get("asset_class") or "UNKNOWN"),
                "sector": str(context.get("sector") or "UNKNOWN"),
                "currency": str(context.get("currency") or "UNKNOWN"),
                "regime": str(context.get("regime") or "UNKNOWN"),
                "feature_timestamp": feature_time,
                "as_of": as_of,
                **numeric,
            }
        )
    return pd.DataFrame(rows), missing


def _counterfactual_status(
    tournament: dict[str, Any], inference: dict[str, Any]
) -> dict[str, Any]:
    actionable = [
        row
        for row in inference["model_evidence"]
        if row["meta_take"] and not row["abstained"] and not row["out_of_distribution"]
    ]
    return {
        "schema": "decision_intelligence_counterfactual_v1",
        "status": tournament["promotion_status"],
        "would_take_count": len(actionable),
        "would_abstain_count": sum(
            bool(row["abstained"]) for row in inference["model_evidence"]
        ),
        "financial_effect_applied": False,
        "reason": "NO_VALIDATED_INCREMENTAL_OOS_AUTHORITY",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        joblib.dump(value, name)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(name, index=False)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _manifest_age_hours(manifest: dict[str, Any]) -> float:
    generated = pd.to_datetime(
        manifest.get("generated_at"), utc=True, errors="coerce"
    )
    return (
        float((pd.Timestamp.now(tz="UTC") - generated).total_seconds() / 3_600)
        if not pd.isna(generated)
        else float("inf")
    )


__all__ = [
    "enqueue_refresh_if_due",
    "infer_current_opportunities",
    "refresh_if_due",
    "refresh_decision_intelligence",
]
