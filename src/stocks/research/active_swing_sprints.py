from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.research.evidence_throughput import (
    publish_evidence_throughput,
)
from stocks.research.rejected_shadow import (
    gate_value_attribution_status,
    rejected_shadow_status,
    settle_rejected_opportunities,
)


PRIVATE_ROOT = Path("data/market_context/private")
OUTPUT_ROOT = Path("output/research/active_swing")
COVERAGE_STATES = frozenset(
    {
        "AVAILABLE_LIVE",
        "AVAILABLE_DELAYED",
        "AVAILABLE_BAR_PROXY",
        "UNAVAILABLE",
        "STALE",
    }
)
TREATMENTS = (
    "BASE",
    "LEVEL1_TAPE",
    "TAPE_DEPTH",
    "ASSET_PROFILE",
)
TERMINAL_TRADES = frozenset(
    {"STOPPED", "TP1_EXIT", "TP2_EXIT", "TRAIL_EXIT", "TIME_EXIT"}
)
ROLE_SPECS = {
    "STRATEGIC_ALLOCATION": {
        "timeframes": {"1d", "1w", "1mo"},
        "minimum_fills": 20,
    },
    "ACTIVE_SWING": {
        "timeframes": {"1h", "2h", "4h", "1d"},
        "minimum_fills": 50,
    },
    "TACTICAL_ENTRY": {
        "timeframes": {"15m", "1h"},
        "minimum_fills": 50,
    },
    "EVENT_DRIVEN": {
        "timeframes": {"1h", "2h", "4h", "1d"},
        "minimum_fills": 30,
    },
    "COMMODITY_PROXY": {
        "timeframes": {"1h", "2h", "4h", "1d", "1w", "1mo"},
        "minimum_fills": 30,
    },
}
EVENT_MARKERS = ("event", "earnings", "gap", "news", "drift")
AUTHORITY = {
    "strategy_authority": "NONE",
    "execution_authority": "NONE",
    "automatic_orders": 0,
    "broker_calls": 0,
    "order_calls": 0,
}
NUMERIC_ML_FEATURES = (
    "setup_score",
    "asset_profile_score",
    "asset_profile_coverage",
    "asset_bias_score",
    "asset_bias_confidence",
    "reward_risk",
    "spread_bps",
)
REGIME_FEATURE = "market_regime"
MINIMUM_LORO_REGIMES = 3
MINIMUM_LORO_REGIME_ROWS = 30
MINIMUM_LORO_TEST_ROWS = 10
MINIMUM_LORO_TRAIN_ROWS = 60


def publish_shortlist_coverage(
    project_root: Path,
    *,
    structural_limit: int = 20,
    tape_limit: int = 10,
    depth_limit: int = 5,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    if not 1 <= structural_limit <= 50:
        raise ValueError("structural_limit must be between 1 and 50")
    if not 0 <= depth_limit <= tape_limit <= structural_limit:
        raise ValueError("require depth_limit <= tape_limit <= structural_limit")
    now = _utc(observed_at or datetime.now(UTC))
    shortlist = _read_json(
        project_root / "output/market_context/entry-shortlist.json"
    )
    observations = [
        row
        for row in shortlist.get("observations", [])
        if isinstance(row, dict)
    ][:structural_limit]
    rows: list[dict[str, Any]] = []
    for position, observation in enumerate(observations, start=1):
        entry = observation.get("entry_snapshot", {})
        tape = entry.get("tape", {}) if isinstance(entry, dict) else {}
        depth = entry.get("depth", {}) if isinstance(entry, dict) else {}
        profile = (
            observation.get("decision_contract", {}).get("asset_profile", {})
        )
        asset_class = str(profile.get("asset_class") or "STOCK").upper()
        tape_requested = position <= tape_limit
        depth_requested = position <= depth_limit
        level1_status = _component_status(tape, now=now, requested=tape_requested)
        tape_status = _observed_component_status(
            tape,
            now=now,
            requested=tape_requested,
            required_data_prefix="OBSERVED_",
        )
        depth_status = _observed_component_status(
            depth,
            now=now,
            requested=depth_requested,
            required_data_prefix="OBSERVED_",
        )
        row = {
            "rank": position,
            "episode_id": observation.get("episode_id"),
            "symbol_fingerprint": stable_hash(
                {"symbol": str(observation.get("symbol") or "").upper()}
            )[:20],
            "asset_class": asset_class,
            "timeframe": observation.get("timeframe"),
            "level1_requested": tape_requested,
            "tape_requested": tape_requested,
            "depth_requested": depth_requested,
            "level1_status": level1_status,
            "tape_status": tape_status,
            "depth_status": depth_status,
            "asset_profile_status": _profile_status(profile),
            "stock_market_sector_relative_strength_status": (
                _profile_feature_status(profile, "relative_strength")
                if asset_class == "STOCK"
                else "UNAVAILABLE"
            ),
            "etf_breadth_status": (
                _profile_feature_status(profile, "breadth")
                if asset_class == "ETF"
                else "UNAVAILABLE"
            ),
            "etf_fair_value_status": (
                _profile_feature_status(profile, "fair_value")
                if asset_class == "ETF"
                else "UNAVAILABLE"
            ),
            "commodity_curve_status": (
                _profile_feature_status(profile, "curve")
                if asset_class == "COMMODITY_PROXY"
                else "UNAVAILABLE"
            ),
            "commodity_future_confirmation_status": (
                _profile_feature_status(profile, "futures_orderflow")
                if asset_class == "COMMODITY_PROXY"
                else "UNAVAILABLE"
            ),
            "bar_proxy_used_as_tape": False,
            "bar_proxy_used_as_depth": False,
        }
        rows.append(row)
    counts = {
        state: sum(
            row[field] == state
            for row in rows
            for field in ("level1_status", "tape_status", "depth_status")
        )
        for state in sorted(COVERAGE_STATES)
    }
    requested_tape = [row for row in rows if row["tape_requested"]]
    requested_depth = [row for row in rows if row["depth_requested"]]
    report = {
        "schema": "active_swing_bounded_shortlist_coverage_v1",
        "status": "GO" if rows else "NO_CURRENT_SETUPS",
        "generated_at": now.isoformat(),
        "funnel": {
            "input_signal_count": int(
                shortlist.get("signal_funnel", {}).get("input_signals", 0)
            ),
            "structural_candidates": len(rows),
            "level1_tape_budget": tape_limit,
            "level1_tape_requested": len(requested_tape),
            "depth_budget": depth_limit,
            "depth_requested": len(requested_depth),
        },
        "coverage_counts": counts,
        "observed_tape_count": sum(
            row["tape_status"] in {"AVAILABLE_LIVE", "AVAILABLE_DELAYED"}
            for row in requested_tape
        ),
        "observed_depth_count": sum(
            row["depth_status"] in {"AVAILABLE_LIVE", "AVAILABLE_DELAYED"}
            for row in requested_depth
        ),
        "bar_proxy_can_claim_tape_or_depth": False,
        "provider_statuses_explicit": True,
        "rows": rows,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    root = project_root / OUTPUT_ROOT / "shortlist_data"
    _write_json(root / "coverage.json", report)
    pd.DataFrame(rows).to_parquet(root / "coverage.parquet", index=False)
    return report


def run_entry_filter_experiment(project_root: Path) -> dict[str, Any]:
    episodes, outcomes = _joined_forward_rows(project_root)
    terminal_outcomes = {
        str(row.get("episode_id")): row
        for row in outcomes
        if row.get("terminal")
    }
    records: list[dict[str, Any]] = []
    for episode in episodes:
        outcome = terminal_outcomes.get(str(episode.get("episode_id")))
        if outcome is None:
            continue
        for treatment in TREATMENTS:
            accepted = _treatment_accepts(episode, treatment)
            records.append(
                _experiment_record(episode, outcome, treatment, accepted)
            )
    frame = pd.DataFrame(records)
    summaries = [
        _treatment_summary(frame, treatment=treatment)
        for treatment in TREATMENTS
    ]
    base = next(row for row in summaries if row["treatment"] == "BASE")
    for row in summaries:
        row["incremental_net_expectancy_R"] = _difference(
            row["net_expectancy_R"], base["net_expectancy_R"]
        )
    closed = int(base["closed_trade_count"])
    status = "GO" if closed >= 50 else "INSUFFICIENT_SAMPLE"
    report = {
        "schema": "active_swing_entry_filter_experiment_v1",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "treatments": summaries,
        "closed_independent_base_episodes": closed,
        "minimum_closed_independent_episodes": 50,
        "promotion_eligible": bool(
            closed >= 50
            and any(
                (row["incremental_net_expectancy_R"] or 0) > 0
                and row["cost_stress_status"] == "POSITIVE"
                for row in summaries
                if row["treatment"] != "BASE"
            )
        ),
        "same_frozen_setups_stops_targets": True,
        "counterfactual_outcomes_reused_without_mutation": True,
        "unfilled_entries_excluded_from_pnl": True,
        "intrabar_ambiguous_excluded": True,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    root = project_root / OUTPUT_ROOT / "entry_filter_experiment"
    _write_json(root / "results.json", report)
    frame.to_parquet(root / "episode-treatment-results.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(root / "treatment-summary.parquet", index=False)
    return report


def publish_active_swing_leaderboards(project_root: Path) -> dict[str, Any]:
    source = project_root / "output/research/phase11_12/strategy-summary.parquet"
    frame = pd.read_parquet(source) if source.is_file() else pd.DataFrame()
    experiment = _read_json(
        project_root / OUTPUT_ROOT / "entry_filter_experiment/results.json"
    )
    reports: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for role, spec in ROLE_SPECS.items():
        candidates = _role_candidates(frame, role=role, spec=spec)
        if role == "TACTICAL_ENTRY":
            candidates = _tactical_candidates(experiment)
        candidates = _rank_role(candidates)
        candidates["operational_designation"] = [
            "CHAMPION" if index == 0 else "CHALLENGER" if index < 3 else "RESEARCH_ONLY"
            for index in range(len(candidates))
        ]
        selected = candidates.head(3).copy()
        rows = selected.replace({np.nan: None}).to_dict("records")
        for row in rows:
            row["role"] = role
        all_rows.extend(rows)
        reports[role] = {
            "status": "GO" if rows else "DATA_UNAVAILABLE",
            "candidate_count": len(candidates),
            "published_count": len(rows),
            "champion_count": sum(
                row["operational_designation"] == "CHAMPION" for row in rows
            ),
            "challenger_count": sum(
                row["operational_designation"] == "CHALLENGER" for row in rows
            ),
            "rows": rows,
        }
    report = {
        "schema": "active_swing_role_leaderboards_v1",
        "status": "GO" if all_rows else "DATA_UNAVAILABLE",
        "generated_at": datetime.now(UTC).isoformat(),
        "roles": reports,
        "rank_weights": {
            "expectancy": 0.25,
            "stability": 0.20,
            "cost_robustness": 0.15,
            "drawdown_quality": 0.15,
            "forward_evidence": 0.15,
            "coverage": 0.10,
        },
        "sample_shrinkage_applied": True,
        "maximum_champions_per_role": 1,
        "maximum_challengers_per_role": 2,
        "cross_role_ranking_allowed": False,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    root = project_root / OUTPUT_ROOT / "leaderboards"
    _write_json(root / "status.json", report)
    pd.DataFrame(all_rows).to_parquet(root / "champions-and-challengers.parquet", index=False)
    return report


def train_selective_ml(project_root: Path) -> dict[str, Any]:
    episodes, outcomes = _joined_forward_rows(project_root)
    joined = _ml_rows(episodes, outcomes)
    label_provenance = _label_provenance_inventory(
        project_root,
        episodes=episodes,
        outcomes=outcomes,
        trainable_rows=joined,
    )
    sample_count = len(joined)
    maturity = (
        "PAPER_RANKING_RESEARCH_ELIGIBLE"
        if sample_count >= 1_000
        else "SHADOW_COMPARISON_ELIGIBLE"
        if sample_count >= 500
        else "EXPERIMENTAL_SMOKE_ELIGIBLE"
        if sample_count >= 100
        else "NOT_TRAINED_INSUFFICIENT_FORWARD_LABELS"
    )
    report: dict[str, Any] = {
        "schema": "active_swing_selective_ml_v2",
        "status": maturity,
        "generated_at": datetime.now(UTC).isoformat(),
        "closed_trainable_episode_count": sample_count,
        "minimum_smoke_episodes": 100,
        "minimum_training_episodes": 500,
        "shadow_comparison_minimum": 500,
        "paper_ranking_minimum": 1_000,
        "targets": [
            "P_TP1_BEFORE_STOP",
            "EXPECTED_NET_R",
            "EXPECTED_MAE",
            "FILL_PROBABILITY",
            "EXPECTED_SLIPPAGE",
        ],
        "output_labels": ["RANK_HIGH", "RANK_NORMAL", "ABSTAIN"],
        "automatic_retraining": False,
        "canonical_label_source": "PHASE9_CANONICAL_BROKER_FILL",
        "canonical_close_required": True,
        "feature_snapshot_hash_match_required": True,
        "bar_simulation_labels_trainable": False,
        "open_episodes_trainable": False,
        "random_train_test_split": False,
        "model_may_only_rank_or_reduce_sizing": True,
        "model_can_raise_risk_caps": False,
        "model_can_widen_stops": False,
        "regime_conditioning": "SINGLE_MODEL_EXPLICIT_REGIME_FEATURE",
        "feature_availability_indicators": True,
        "leave_one_regime_out_validation": True,
        "mixture_of_experts_status": "DEFERRED_INSUFFICIENT_INDEPENDENT_LABELS",
        "mixture_of_experts_minimum_labels": 3_000,
        "reinforcement_learning_status": "DISABLED_PREMATURE_SAMPLE_SIZE",
        "reinforcement_learning_authority": "NONE",
        "regime_dataset": _regime_dataset_summary(joined),
        "label_provenance": label_provenance,
        "model_authority": "NONE",
        "order_advice_authority": "NONE",
        **AUTHORITY,
    }
    root = project_root / OUTPUT_ROOT / "selective_ml"
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "label-provenance.json", label_provenance)
    if sample_count < 100:
        report["trained"] = False
        report["blockers"] = ["INSUFFICIENT_CLOSED_FORWARD_LABELS"]
        report["regime_generalization"] = _leave_one_regime_out(joined)
        report["content_hash"] = stable_hash(report)
        _write_json(root / "status.json", report)
        _write_json(
            root / "regime-generalization.json",
            report["regime_generalization"],
        )
        return report
    fit = _fit_models(joined, root=root)
    report.update(fit)
    report["trained"] = bool(fit.get("trained", False))
    report["regime_generalization"] = _leave_one_regime_out(joined)
    report["content_hash"] = stable_hash(report)
    _write_json(root / "status.json", report)
    _write_json(
        root / "regime-generalization.json",
        report["regime_generalization"],
    )
    return report


def active_swing_sprint_status(project_root: Path) -> dict[str, Any]:
    paths = {
        "shortlist_data": project_root / OUTPUT_ROOT / "shortlist_data/coverage.json",
        "entry_filter_experiment": project_root / OUTPUT_ROOT / "entry_filter_experiment/results.json",
        "leaderboards": project_root / OUTPUT_ROOT / "leaderboards/status.json",
        "selective_ml": project_root / OUTPUT_ROOT / "selective_ml/status.json",
        "rejected_shadow": (
            project_root / OUTPUT_ROOT / "rejected_shadow/status.json"
        ),
        "evidence_throughput": (
            project_root
            / "output/research/evidence_throughput/status.json"
        ),
    }
    components = {
        name: _read_json(path) if path.is_file() else {"status": "NOT_RUN"}
        for name, path in paths.items()
    }
    report = {
        "schema": "active_swing_sprints_3_6_status_v1",
        "status": (
            "GO"
            if all(value.get("status") != "NOT_RUN" for value in components.values())
            else "PARTIAL"
        ),
        "components": {
            name: value.get("status") for name, value in components.items()
        },
        "live_tape_depth_missing_is_explicit": True,
        "financial_promotion_requires_forward_sample": True,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    _write_json(project_root / OUTPUT_ROOT / "status.json", report)
    return report


def run_active_swing_sprints(project_root: Path) -> dict[str, Any]:
    publish_shortlist_coverage(project_root)
    settle_rejected_opportunities(project_root)
    run_entry_filter_experiment(project_root)
    publish_active_swing_leaderboards(project_root)
    train_selective_ml(project_root)
    publish_evidence_throughput(project_root)
    return active_swing_sprint_status(project_root)


def refresh_active_swing_observation(project_root: Path) -> dict[str, Any]:
    """Refresh deterministic observation artifacts without retraining ML."""
    publish_shortlist_coverage(project_root)
    settle_rejected_opportunities(project_root)
    run_entry_filter_experiment(project_root)
    publish_active_swing_leaderboards(project_root)
    publish_evidence_throughput(project_root)
    ml_status = project_root / OUTPUT_ROOT / "selective_ml/status.json"
    if not ml_status.is_file():
        report = {
            "schema": "active_swing_selective_ml_v2",
            "status": "NOT_TRAINED_INSUFFICIENT_FORWARD_LABELS",
            "trained": False,
            "automatic_retraining": False,
            "model_authority": "NONE",
            **AUTHORITY,
        }
        report["content_hash"] = stable_hash(report)
        _write_json(ml_status, report)
    return active_swing_sprint_status(project_root)


def _component_status(
    component: Mapping[str, Any], *, now: datetime, requested: bool
) -> str:
    if not requested:
        return "UNAVAILABLE"
    status = str(component.get("status") or "").upper()
    if "UNAVAILABLE" in status or not component:
        return "UNAVAILABLE"
    if _is_stale(component, now=now):
        return "STALE"
    data_class = str(component.get("data_class") or "").upper()
    if "BAR_FLOW_PROXY" in data_class:
        return "AVAILABLE_BAR_PROXY"
    if "DELAYED" in data_class:
        return "AVAILABLE_DELAYED"
    return "AVAILABLE_LIVE"


def _observed_component_status(
    component: Mapping[str, Any],
    *,
    now: datetime,
    requested: bool,
    required_data_prefix: str,
) -> str:
    status = _component_status(component, now=now, requested=requested)
    data_class = str(component.get("data_class") or "").upper()
    if status == "AVAILABLE_BAR_PROXY":
        return "UNAVAILABLE"
    if status in {"AVAILABLE_LIVE", "AVAILABLE_DELAYED"} and not data_class.startswith(
        required_data_prefix
    ):
        return "UNAVAILABLE"
    return status


def _profile_status(profile: Mapping[str, Any]) -> str:
    if not profile:
        return "UNAVAILABLE"
    if float(profile.get("data_coverage_ratio") or 0.0) >= 0.8:
        return "AVAILABLE_LIVE"
    return "AVAILABLE_BAR_PROXY"


def _profile_feature_status(profile: Mapping[str, Any], name: str) -> str:
    value = profile.get("component_scores", {}).get(name)
    return "AVAILABLE_BAR_PROXY" if _finite(value) else "UNAVAILABLE"


def _is_stale(component: Mapping[str, Any], *, now: datetime) -> bool:
    for key in ("observed_at", "timestamp", "last_timestamp", "captured_at"):
        value = component.get(key)
        if value:
            try:
                timestamp = pd.Timestamp(value)
                timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
                return now - timestamp.to_pydatetime() > pd.Timedelta(minutes=15)
            except (TypeError, ValueError):
                return True
    return False


def _joined_forward_rows(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = [
        row
        for row in _read_jsonl(project_root / PRIVATE_ROOT / "entry-episodes.jsonl")
        if row.get("schema") == "active_swing_forward_episode_v1"
        and row.get("feature_snapshot_hash")
    ]
    outcomes = [
        row
        for row in _read_jsonl(project_root / PRIVATE_ROOT / "entry-episode-outcomes.jsonl")
        if row.get("schema") == "active_swing_forward_episode_outcome_v1"
    ]
    revisions = [
        row
        for row in _read_jsonl(
            project_root
            / PRIVATE_ROOT
            / "entry-episode-outcome-revisions.jsonl"
        )
        if row.get("schema") == "active_swing_forward_episode_outcome_v1"
        and row.get("research_revision_accepted") is True
    ]
    effective = {
        str(row.get("episode_id")): row
        for row in outcomes
        if row.get("episode_id")
    }
    effective.update(
        {
            str(row.get("episode_id")): row
            for row in revisions
            if row.get("episode_id")
        }
    )
    return episodes, list(effective.values())


def _treatment_accepts(episode: Mapping[str, Any], treatment: str) -> bool:
    decision = episode.get("decision_contract", {})
    gates = decision.get("gates", {})
    if not bool(
        decision.get("hard_veto_pass")
        or decision.get("research_observation_eligible")
    ):
        return False
    if treatment == "BASE":
        return True
    if treatment == "LEVEL1_TAPE":
        return bool(gates.get("observed_tape_available") and gates.get("observed_tape_confirms"))
    if treatment == "TAPE_DEPTH":
        return bool(
            gates.get("observed_tape_available")
            and gates.get("observed_tape_confirms")
            and gates.get("observed_depth_available")
            and gates.get("observed_depth_confirms")
        )
    profile = decision.get("asset_profile", {})
    return bool(float(profile.get("data_coverage_ratio") or 0.0) >= 0.6 and float(profile.get("coverage_adjusted_score") or 0.0) >= 55.0)


def _experiment_record(
    episode: Mapping[str, Any],
    outcome: Mapping[str, Any],
    treatment: str,
    accepted: bool,
) -> dict[str, Any]:
    status = str(outcome.get("terminal_status") or "")
    return {
        "episode_id": episode.get("episode_id"),
        "treatment": treatment,
        "accepted": bool(accepted),
        "would_fill": bool(outcome.get("would_fill")),
        "closed_trade": status in TERMINAL_TRADES,
        "terminal_status": status,
        "net_R": _number_or_none(outcome.get("net_R")),
        "maximum_adverse_excursion": _number_or_none(outcome.get("maximum_adverse_excursion")),
        "maximum_favourable_excursion": _number_or_none(outcome.get("maximum_favourable_excursion")),
        "slippage_bps": _number_or_none(outcome.get("slippage_from_decision_bps")),
        "asset_class": outcome.get("asset_class"),
        "market_regime": episode.get("decision_contract", {}).get("market_regime"),
        "positive_outcome": bool((_number_or_none(outcome.get("net_R")) or 0.0) > 0),
    }


def _treatment_summary(frame: pd.DataFrame, *, treatment: str) -> dict[str, Any]:
    if frame.empty:
        selected = frame
    else:
        selected = frame.loc[(frame["treatment"] == treatment) & frame["accepted"]]
    trades = selected.loc[selected.get("closed_trade", pd.Series(dtype=bool)).fillna(False)] if not selected.empty else selected
    pnl = pd.to_numeric(trades.get("net_R", pd.Series(dtype=float)), errors="coerce").dropna()
    gains = float(pnl.loc[pnl > 0].sum())
    losses = float(pnl.loc[pnl < 0].sum())
    pf = gains / abs(losses) if losses < 0 else math.inf if gains > 0 else None
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax() if not equity.empty else pd.Series(dtype=float)
    all_treatment = frame.loc[frame["treatment"] == treatment] if not frame.empty else frame
    rejected = (
        all_treatment.loc[
            (~all_treatment["accepted"]) & all_treatment["closed_trade"]
        ]
        if not all_treatment.empty
        else all_treatment
    )
    return {
        "treatment": treatment,
        "terminal_episode_count": len(all_treatment),
        "accepted_episode_count": len(selected),
        "closed_trade_count": len(trades),
        "fill_ratio": round(float(selected["would_fill"].mean()), 8) if not selected.empty else None,
        "net_expectancy_R": round(float(pnl.mean()), 8) if not pnl.empty else None,
        "median_net_R": round(float(pnl.median()), 8) if not pnl.empty else None,
        "profit_factor": None if pf is None or math.isinf(pf) else round(pf, 8),
        "profit_factor_status": "PERFECT_NO_LOSSES" if pf is not None and math.isinf(pf) else "DEFINED" if pf is not None else "NO_CLOSED_TRADES",
        "maximum_drawdown_R": round(float(drawdown.min()), 8) if not drawdown.empty else None,
        "median_MAE_R": _median(trades.get("maximum_adverse_excursion", pd.Series(dtype=float))),
        "median_MFE_R": _median(trades.get("maximum_favourable_excursion", pd.Series(dtype=float))),
        "average_slippage_bps": _mean(trades.get("slippage_bps", pd.Series(dtype=float))),
        "missed_winner_count": int(rejected.get("positive_outcome", pd.Series(dtype=bool)).sum()) if not rejected.empty else 0,
        "avoided_loser_count": int((~rejected.get("positive_outcome", pd.Series(dtype=bool))).sum()) if not rejected.empty else 0,
        "asset_class_count": int(trades.get("asset_class", pd.Series(dtype=object)).nunique()),
        "regime_count": int(trades.get("market_regime", pd.Series(dtype=object)).nunique()),
        "cost_stress_status": "POSITIVE" if not pnl.empty and float(pnl.mean()) > 0 else "NOT_PROVEN",
    }


def _role_candidates(frame: pd.DataFrame, *, role: str, spec: Mapping[str, Any]) -> pd.DataFrame:
    if frame.empty or role == "TACTICAL_ENTRY":
        return pd.DataFrame()
    work = frame.copy()
    for column in ("CAGR", "Sharpe", "period_profit_factor", "stress_50bps_profit_factor", "maximum_drawdown", "fill_count"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    gate = (
        work["status"].eq("COMPLETE")
        & work["timeframe"].isin(spec["timeframes"])
        & work["fill_count"].ge(spec["minimum_fills"])
        & work["CAGR"].gt(0.0)
        & work["period_profit_factor"].gt(1.0)
        & work["stress_50bps_profit_factor"].gt(1.0)
        & work["maximum_drawdown"].ge(-0.35)
    )
    if role == "EVENT_DRIVEN":
        gate &= work["formula"].astype(str).str.lower().apply(lambda value: any(marker in value for marker in EVENT_MARKERS))
    if role == "COMMODITY_PROXY":
        gate &= work["asset_class"].eq("COMMODITY_PROXY")
    selected = work.loc[gate].copy()
    if selected.empty:
        return selected
    selected["candidate_id"] = selected["strategy_id"]
    selected["expectancy"] = selected["period_profit_factor"] - 1.0
    selected["stability"] = selected["Sharpe"]
    selected["cost_robustness"] = selected["stress_50bps_profit_factor"] - 1.0
    selected["drawdown_quality"] = -selected["maximum_drawdown"].abs()
    selected["forward_evidence"] = 0.0
    selected["coverage"] = np.log1p(selected["fill_count"])
    return selected


def _tactical_candidates(experiment: Mapping[str, Any]) -> pd.DataFrame:
    if (
        str(experiment.get("status")) != "GO"
        or int(experiment.get("closed_independent_base_episodes") or 0) < 50
    ):
        return pd.DataFrame()
    rows = []
    for row in experiment.get("treatments", []):
        if row.get("treatment") == "BASE":
            continue
        rows.append(
            {
                "candidate_id": "ENTRY_FILTER_" + str(row.get("treatment")),
                "strategy_id": "ENTRY_FILTER_" + str(row.get("treatment")),
                "formula": row.get("treatment"),
                "timeframe": "1h",
                "asset_class": "MULTI_ASSET",
                "fill_count": row.get("closed_trade_count", 0),
                "expectancy": row.get("incremental_net_expectancy_R") or 0.0,
                "stability": 0.0,
                "cost_robustness": 1.0 if row.get("cost_stress_status") == "POSITIVE" else 0.0,
                "drawdown_quality": -(abs(row.get("maximum_drawdown_R") or 0.0)),
                "forward_evidence": row.get("closed_trade_count", 0),
                "coverage": row.get("accepted_episode_count", 0),
            }
        )
    return pd.DataFrame(rows)


def _rank_role(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    components = {
        "expectancy": 0.25,
        "stability": 0.20,
        "cost_robustness": 0.15,
        "drawdown_quality": 0.15,
        "forward_evidence": 0.15,
        "coverage": 0.10,
    }
    score = pd.Series(0.0, index=work.index)
    for column, weight in components.items():
        values = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        score += weight * values.rank(pct=True, method="average")
    sample = pd.to_numeric(work.get("fill_count", 0), errors="coerce").fillna(0.0)
    shrinkage = sample / (sample + 100.0)
    work["unshrunk_rank_score"] = score
    work["sample_shrinkage"] = shrinkage
    work["rank_score"] = 0.5 + shrinkage * (score - 0.5)
    return work.sort_values(["rank_score", "candidate_id"], ascending=[False, True])


def _ml_rows(episodes: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> pd.DataFrame:
    episode_map = {str(row.get("episode_id")): row for row in episodes}
    rows = []
    for outcome in outcomes:
        episode = episode_map.get(str(outcome.get("episode_id")))
        if _canonical_ml_exclusion_reason(outcome, episode) is not None:
            continue
        assert episode is not None
        rows.append(_ml_feature_record(episode, outcome))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["decision_timestamp", "target"]).sort_values("decision_timestamp")
    return frame


def _canonical_ml_exclusion_reason(
    outcome: Mapping[str, Any],
    episode: Mapping[str, Any] | None,
) -> str | None:
    if outcome.get("label_source") != "PHASE9_CANONICAL_BROKER_FILL":
        return "NON_CANONICAL_LABEL_SOURCE"
    if outcome.get("canonical_close") is not True:
        return "CANONICAL_CLOSE_MISSING"
    if outcome.get("canonical_fill_evidence") is not True:
        return "CANONICAL_FILL_EVIDENCE_MISSING"
    if str(outcome.get("terminal_status")) not in TERMINAL_TRADES:
        return "NON_TRADE_TERMINAL_STATUS"
    if episode is None:
        return "ORPHAN_OUTCOME"
    episode_hash = str(episode.get("feature_snapshot_hash") or "")
    outcome_hash = str(outcome.get("feature_snapshot_hash") or "")
    if not episode_hash or outcome_hash != episode_hash:
        return "FEATURE_SNAPSHOT_HASH_MISMATCH"
    decision_timestamp = pd.to_datetime(
        episode.get("decision_timestamp"),
        utc=True,
        errors="coerce",
    )
    if pd.isna(decision_timestamp):
        return "DECISION_TIMESTAMP_INVALID"
    for field in ("fill_timestamp", "exit_timestamp"):
        value = outcome.get(field)
        if not value:
            continue
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(timestamp) or timestamp < decision_timestamp:
            return "OUTCOME_TIMESTAMP_NOT_CAUSAL"
    return None


def _ml_feature_record(
    episode: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    decision = episode.get("decision_contract", {})
    profile = decision.get("asset_profile", {})
    context = episode.get("context_snapshot", {})
    setup = episode.get("setup_snapshot", {})
    return {
        "decision_timestamp": episode.get("decision_timestamp"),
        "setup_score": _number_or_none(decision.get("setup_score_within_family")),
        "asset_profile_score": _number_or_none(
            profile.get("coverage_adjusted_score")
        ),
        "asset_profile_coverage": _number_or_none(
            profile.get("data_coverage_ratio")
        ),
        "asset_bias_score": _number_or_none(context.get("asset_bias_score")),
        "asset_bias_confidence": _number_or_none(
            context.get("asset_bias_confidence")
        ),
        "reward_risk": _number_or_none(setup.get("reward_risk_1")),
        "spread_bps": _number_or_none(outcome.get("spread_at_decision_bps")),
        "market_regime": str(
            decision.get("market_regime") or "UNAVAILABLE_AT_DECISION"
        ).upper(),
        "target": int(
            str(outcome.get("terminal_status")) in {"TP1_EXIT", "TP2_EXIT"}
        ),
        "net_R": _number_or_none(outcome.get("net_R")),
        "mae": _number_or_none(outcome.get("maximum_adverse_excursion")),
        "slippage_bps": _number_or_none(
            outcome.get("slippage_from_decision_bps")
        ),
        "label_source": outcome.get("label_source"),
    }


def _label_provenance_inventory(
    project_root: Path,
    *,
    episodes: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    trainable_rows: pd.DataFrame,
) -> dict[str, Any]:
    episode_map = {str(row.get("episode_id")): row for row in episodes}
    exclusions = Counter(
        reason
        for outcome in outcomes
        if (
            reason := _canonical_ml_exclusion_reason(
                outcome,
                episode_map.get(str(outcome.get("episode_id"))),
            )
        )
    )
    terminal_outcomes = [row for row in outcomes if row.get("terminal") is True]
    terminal_trades = [
        row
        for row in terminal_outcomes
        if str(row.get("terminal_status")) in TERMINAL_TRADES
    ]
    feature_rows = [
        _ml_feature_record(episode, outcome)
        for outcome in terminal_outcomes
        if (
            episode := episode_map.get(str(outcome.get("episode_id")))
        )
        is not None
    ]
    full_feature_count = sum(
        all(row.get(feature) is not None for feature in NUMERIC_ML_FEATURES)
        and row.get(REGIME_FEATURE) != "UNAVAILABLE_AT_DECISION"
        for row in feature_rows
    )
    historical = _historical_research_cohort(project_root)
    news = _news_label_cohort(project_root)
    paper = _execution_label_cohort(
        project_root / "data/execution/phase9/private/paper_execution.sqlite3",
        environment="PAPER",
    )
    live = _execution_label_cohort(
        project_root / "data/execution/live/private/live_execution.sqlite3",
        environment="LIVE",
    )
    cohorts = {
        "HISTORICAL_BACKTEST_RESEARCH": historical,
        "FORWARD_COUNTERFACTUAL_OBSERVATION": {
            "observed_episode_count": len(episodes),
            "terminal_outcome_count": len(terminal_outcomes),
            "closed_trade_outcome_count": len(terminal_trades),
            "full_feature_row_count": full_feature_count,
            "full_feature_coverage_ratio": round(
                full_feature_count / len(feature_rows), 8
            )
            if feature_rows
            else None,
            "label_sources": dict(
                sorted(
                    Counter(
                        str(row.get("label_source") or "UNSPECIFIED")
                        for row in terminal_outcomes
                    ).items()
                )
            ),
            "trainable_for_active_swing_meta_label": False,
            "reason": "COUNTERFACTUAL_OR_NONCANONICAL_FORWARD_OUTCOMES",
        },
        "PAPER_BROKER_EXECUTION": paper,
        "LIVE_BROKER_EXECUTION": live,
        "NEWS_EVENT_CAR": news,
        "ACTIVE_SWING_CANONICAL_CLOSED": {
            "observed_label_count": len(trainable_rows),
            "trainable_for_active_swing_meta_label": True,
            "label_source": "PHASE9_CANONICAL_BROKER_FILL",
            "feature_snapshot_hash_match_required": True,
        },
    }
    report = {
        "schema": "active_swing_label_provenance_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "GO",
        "cohorts": cohorts,
        "canonical_trainable_label_count": len(trainable_rows),
        "canonical_exclusion_reasons": dict(sorted(exclusions.items())),
        "cohort_isolation": True,
        "historical_backtest_labels_are_execution_evidence": False,
        "counterfactual_forward_labels_are_broker_fill_evidence": False,
        "paper_labels_are_live_labels": False,
        "news_labels_are_trade_meta_labels": False,
        "open_episodes_trainable": False,
        "automatic_promotion": False,
        "model_authority": "NONE",
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _historical_research_cohort(project_root: Path) -> dict[str, Any]:
    paths = (
        Path("output/research/phase11_7/selected-closed-episodes.csv"),
        Path("output/research/phase11_7/rotation/selected-closed-episodes.csv"),
    )
    sources = []
    observed_count = 0
    for relative in paths:
        path = project_root / relative
        count = 0
        if path.is_file():
            try:
                count = len(pd.read_csv(path))
            except (OSError, ValueError, pd.errors.ParserError):
                count = 0
        observed_count += count
        sources.append({"path": relative.as_posix(), "row_count": count})
    return {
        "observed_label_count": observed_count,
        "sources": sources,
        "target_domain": "HISTORICAL_PORTFOLIO_EPISODE_PNL",
        "trainable_for_active_swing_meta_label": False,
        "reason": "NO_POINT_IN_TIME_ACTIVE_SWING_FEATURE_SCHEMA_PARITY",
    }


def _news_label_cohort(project_root: Path) -> dict[str, Any]:
    relative = Path("output/news/event_study/event-labels.parquet")
    path = project_root / relative
    if not path.is_file():
        return {
            "status": "SOURCE_MISSING",
            "observed_label_count": 0,
            "trainable_for_active_swing_meta_label": False,
        }
    try:
        frame = pd.read_parquet(
            path,
            columns=["label_status", "training_eligible", "event_time_mode"],
        )
    except (OSError, ValueError):
        return {
            "status": "SOURCE_UNREADABLE",
            "observed_label_count": 0,
            "trainable_for_active_swing_meta_label": False,
        }
    complete = frame["label_status"].eq("COMPLETE")
    causal = frame["event_time_mode"].eq("OPERATIONAL_CAUSAL")
    eligible = frame["training_eligible"].fillna(False).astype(bool)
    return {
        "status": "GO",
        "source": relative.as_posix(),
        "observed_label_count": int(complete.sum()),
        "operational_causal_complete_count": int((complete & causal).sum()),
        "news_model_training_eligible_count": int(eligible.sum()),
        "target_domain": "NEWS_CUMULATIVE_ABNORMAL_RETURN",
        "trainable_for_active_swing_meta_label": False,
        "reason": "SEPARATE_TARGET_DOMAIN",
    }


def _execution_label_cohort(path: Path, *, environment: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "SOURCE_MISSING",
            "environment": environment,
            "execution_count": 0,
            "commission_count": 0,
            "round_trip_count": 0,
            "strategy_linked_execution_count": 0,
            "trainable_for_active_swing_meta_label": False,
        }
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            execution_payloads = [
                json.loads(str(row[0]))
                for row in connection.execute("SELECT payload_json FROM executions")
            ]
            commission_count = int(
                connection.execute("SELECT COUNT(*) FROM commissions").fetchone()[0]
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError):
        return {
            "status": "SOURCE_UNREADABLE",
            "environment": environment,
            "execution_count": 0,
            "commission_count": 0,
            "round_trip_count": 0,
            "strategy_linked_execution_count": 0,
            "trainable_for_active_swing_meta_label": False,
        }
    quantities: dict[tuple[int, str], dict[str, float]] = {}
    strategy_linked = 0
    for payload in execution_payloads:
        con_id = int(payload.get("con_id") or 0)
        symbol = str(payload.get("symbol") or "").upper()
        side = str(payload.get("side") or "").upper()
        quantity = _number_or_none(payload.get("quantity")) or 0.0
        bucket = quantities.setdefault((con_id, symbol), {"BUY": 0.0, "SELL": 0.0})
        if side in bucket:
            bucket[side] += quantity
        if payload.get("strategy_id") and payload.get("signal_id"):
            strategy_linked += 1
    round_trips = sum(
        1
        for quantity in quantities.values()
        if quantity["BUY"] > 0 and quantity["SELL"] >= quantity["BUY"]
    )
    return {
        "status": "GO",
        "environment": environment,
        "execution_count": len(execution_payloads),
        "commission_count": commission_count,
        "round_trip_count": round_trips,
        "strategy_linked_execution_count": strategy_linked,
        "trainable_for_active_swing_meta_label": False,
        "reason": "CANONICAL_EPISODE_JOIN_AND_CLOSE_REQUIRED",
    }


def _fit_models(frame: pd.DataFrame, *, root: Path) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    split = max(2, int(len(frame) * 0.8))
    purge_count = max(1, min(20, int(len(frame) * 0.02)))
    train_end = max(1, split - purge_count)
    train, test = frame.iloc[:train_end], frame.iloc[split:]
    features = [*NUMERIC_ML_FEATURES, REGIME_FEATURE]
    x_train, y_train = train[features], train["target"]
    x_test, y_test = test[features], test["target"]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return {
            "trained": False,
            "status": "NOT_TRAINED_TARGET_CLASS_INSUFFICIENT",
            "blockers": ["TARGET_CLASS_INSUFFICIENT"],
        }
    models = {
        "LOGISTIC_REGRESSION": _regime_conditioned_pipeline(
            LogisticRegression(max_iter=2_000)
        ),
        "HIST_GRADIENT_BOOSTING": _regime_conditioned_pipeline(
            HistGradientBoostingClassifier(max_iter=100, random_state=42)
        ),
    }
    metrics = {}
    predictions = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        metrics[name] = {
            "roc_auc": round(float(roc_auc_score(y_test, probability)), 8),
            "log_loss": round(float(log_loss(y_test, probability)), 8),
            "brier_score": round(
                float(brier_score_loss(y_test, probability)), 8
            ),
            "calibration_error": _calibration_error(y_test, probability),
        }
        if name == "LOGISTIC_REGRESSION":
            predictions = ["RANK_HIGH" if p >= 0.65 else "ABSTAIN" if p <= 0.35 else "RANK_NORMAL" for p in probability]
    output = test.loc[:, ["decision_timestamp"]].copy()
    output["label"] = predictions
    output.to_parquet(root / "chronological-holdout-predictions.parquet", index=False)
    return {
        "trained": True,
        "chronological_split": True,
        "purged_time_series_split": True,
        "purged_episode_count": purge_count,
        "train_count": len(train),
        "test_count": len(test),
        "features": features,
        "regime_conditioned_single_model": True,
        "feature_availability_indicators": True,
        "model_metrics": metrics,
        "predictions_are_observation_only": True,
        "model_authority": "NONE",
        "risk_caps_can_only_decrease": True,
    }


def _regime_conditioned_pipeline(classifier: Any) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformer = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(add_indicator=True),
                    StandardScaler(),
                ),
                list(NUMERIC_ML_FEATURES),
            ),
            (
                "regime",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                [REGIME_FEATURE],
            ),
        ],
        sparse_threshold=0.0,
    )
    return make_pipeline(transformer, classifier)


def _regime_dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or REGIME_FEATURE not in frame:
        return {
            "status": "INSUFFICIENT_CANONICAL_LABELS",
            "regime_count": 0,
            "available_regime_count": 0,
            "rows_by_regime": {},
        }
    counts = {
        str(key): int(value)
        for key, value in frame[REGIME_FEATURE].value_counts().items()
    }
    available = {
        key: value
        for key, value in counts.items()
        if key != "UNAVAILABLE_AT_DECISION"
    }
    return {
        "status": (
            "GO"
            if len(available) >= MINIMUM_LORO_REGIMES
            else "INSUFFICIENT_REGIME_DIVERSITY"
        ),
        "regime_count": len(counts),
        "available_regime_count": len(available),
        "rows_by_regime": counts,
        "unavailable_regime_row_count": counts.get(
            "UNAVAILABLE_AT_DECISION", 0
        ),
        "missing_values_are_not_zero_filled": True,
        "feature_availability_indicators": True,
    }


def _leave_one_regime_out(frame: pd.DataFrame) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    policy = {
        "minimum_regimes": MINIMUM_LORO_REGIMES,
        "minimum_rows_per_regime": MINIMUM_LORO_REGIME_ROWS,
        "minimum_test_rows": MINIMUM_LORO_TEST_ROWS,
        "minimum_train_rows": MINIMUM_LORO_TRAIN_ROWS,
        "test_selection": "FINAL_20_PERCENT_OF_HELD_OUT_REGIME",
        "training_selection": "OTHER_REGIMES_STRICTLY_BEFORE_TEST_START",
        "purge_policy": "LAST_2_PERCENT_OF_ELIGIBLE_TRAIN_ROWS",
        "objective": "0.60_MEAN_AUC_PLUS_0.40_WORST_AUC_MINUS_0.50_STD_AUC",
    }
    if frame.empty or REGIME_FEATURE not in frame:
        return _blocked_regime_generalization(
            "INSUFFICIENT_CANONICAL_LABELS", policy
        )
    work = frame.loc[
        frame[REGIME_FEATURE].ne("UNAVAILABLE_AT_DECISION")
    ].sort_values("decision_timestamp")
    regimes = sorted(work[REGIME_FEATURE].dropna().astype(str).unique())
    if len(regimes) < MINIMUM_LORO_REGIMES:
        return _blocked_regime_generalization(
            "INSUFFICIENT_REGIME_DIVERSITY", policy,
            observed_regimes=regimes,
        )
    features = [*NUMERIC_ML_FEATURES, REGIME_FEATURE]
    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for regime in regimes:
        held_out = work.loc[work[REGIME_FEATURE].eq(regime)]
        if len(held_out) < MINIMUM_LORO_REGIME_ROWS:
            blocked.append(
                {"regime": regime, "reason": "INSUFFICIENT_REGIME_ROWS"}
            )
            continue
        test_count = max(MINIMUM_LORO_TEST_ROWS, int(len(held_out) * 0.2))
        test = held_out.tail(test_count)
        test_start = test["decision_timestamp"].min()
        train = work.loc[
            work["decision_timestamp"].lt(test_start)
            & work[REGIME_FEATURE].ne(regime)
        ]
        purge_count = max(1, min(20, int(len(train) * 0.02)))
        train = train.iloc[:-purge_count] if len(train) > purge_count else train.iloc[0:0]
        if len(train) < MINIMUM_LORO_TRAIN_ROWS:
            blocked.append(
                {"regime": regime, "reason": "INSUFFICIENT_CAUSAL_TRAIN_ROWS"}
            )
            continue
        if train["target"].nunique() < 2 or test["target"].nunique() < 2:
            blocked.append(
                {"regime": regime, "reason": "TARGET_CLASS_INSUFFICIENT"}
            )
            continue
        model = _regime_conditioned_pipeline(
            LogisticRegression(max_iter=2_000)
        )
        model.fit(train[features], train["target"])
        probability = model.predict_proba(test[features])[:, 1]
        rows.append(
            {
                "held_out_regime": regime,
                "train_count": len(train),
                "test_count": len(test),
                "train_end": train["decision_timestamp"].max().isoformat(),
                "test_start": test_start.isoformat(),
                "causal_ordering_pass": bool(
                    train["decision_timestamp"].max() < test_start
                ),
                "roc_auc": round(
                    float(roc_auc_score(test["target"], probability)), 8
                ),
                "log_loss": round(
                    float(log_loss(test["target"], probability)), 8
                ),
                "brier_score": round(
                    float(brier_score_loss(test["target"], probability)), 8
                ),
                "calibration_error": _calibration_error(
                    test["target"], probability
                ),
            }
        )
    if not rows:
        return _blocked_regime_generalization(
            "NO_EVALUABLE_HELD_OUT_REGIME", policy,
            observed_regimes=regimes,
            blocked=blocked,
        )
    auc = np.asarray([float(row["roc_auc"]) for row in rows])
    mean_auc = float(auc.mean())
    std_auc = float(auc.std(ddof=0))
    worst_index = int(auc.argmin())
    worst_auc = float(auc[worst_index])
    robust_score = 0.60 * mean_auc + 0.40 * worst_auc - 0.50 * std_auc
    return {
        "schema": "active_swing_regime_generalization_v1",
        "status": (
            "GO"
            if len(rows) >= MINIMUM_LORO_REGIMES
            else "PARTIAL_INSUFFICIENT_EVALUABLE_REGIMES"
        ),
        "policy": policy,
        "observed_regimes": regimes,
        "evaluable_regime_count": len(rows),
        "blocked_regimes": blocked,
        "per_regime": rows,
        "mean_auc": round(mean_auc, 8),
        "regime_auc_std": round(std_auc, 8),
        "worst_regime": rows[worst_index]["held_out_regime"],
        "worst_regime_auc": round(worst_auc, 8),
        "robust_generalization_score": round(robust_score, 8),
        "all_folds_causally_ordered": all(
            bool(row["causal_ordering_pass"]) for row in rows
        ),
        "selection_authority": "RESEARCH_SHADOW_ONLY",
        "model_authority": "NONE",
        "automatic_retraining": False,
        "mixture_of_experts_trained": False,
        "reinforcement_learning_trained": False,
        **AUTHORITY,
    }


def _blocked_regime_generalization(
    reason: str,
    policy: Mapping[str, Any],
    *,
    observed_regimes: Iterable[str] = (),
    blocked: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schema": "active_swing_regime_generalization_v1",
        "status": "NOT_EVALUABLE",
        "reason": reason,
        "policy": dict(policy),
        "observed_regimes": list(observed_regimes),
        "evaluable_regime_count": 0,
        "blocked_regimes": [dict(row) for row in blocked],
        "all_folds_causally_ordered": None,
        "selection_authority": "RESEARCH_SHADOW_ONLY",
        "model_authority": "NONE",
        "automatic_retraining": False,
        "mixture_of_experts_trained": False,
        "reinforcement_learning_trained": False,
        **AUTHORITY,
    }


def _calibration_error(target: Any, probability: Any) -> float:
    target_series = pd.Series(target, dtype=float).reset_index(drop=True)
    probability_series = pd.Series(probability, dtype=float).reset_index(drop=True)
    bins = pd.cut(
        probability_series,
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
    )
    error = 0.0
    total = len(target_series)
    if total == 0:
        return math.nan
    for interval in bins.dropna().unique():
        mask = bins.eq(interval)
        if not bool(mask.any()):
            continue
        error += float(mask.mean()) * abs(
            float(target_series.loc[mask].mean())
            - float(probability_series.loc[mask].mean())
        )
    return round(error, 8)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite(value: Any) -> bool:
    return _number_or_none(value) is not None


def _median(values: Iterable[Any]) -> float | None:
    numbers = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return round(float(numbers.median()), 8) if not numbers.empty else None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return round(float(numbers.mean()), 8) if not numbers.empty else None


def _difference(left: Any, right: Any) -> float | None:
    left_number, right_number = _number_or_none(left), _number_or_none(right)
    return round(left_number - right_number, 8) if left_number is not None and right_number is not None else None


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "active_swing_sprint_status",
    "gate_value_attribution_status",
    "publish_active_swing_leaderboards",
    "publish_shortlist_coverage",
    "refresh_active_swing_observation",
    "rejected_shadow_status",
    "run_active_swing_sprints",
    "run_entry_filter_experiment",
    "settle_rejected_opportunities",
    "train_selective_ml",
]
