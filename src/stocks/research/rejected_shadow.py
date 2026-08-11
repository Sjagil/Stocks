from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from filelock import FileLock
from scipy.stats import beta as beta_distribution

from stocks.context.episode_outcomes import _evaluate_episode
from stocks.execution.idempotency import stable_hash
from stocks.research.phase11_9 import _load_current_frames


PRIVATE_EPISODES = Path("data/market_context/private/entry-episodes.jsonl")
PRIVATE_OUTCOMES = Path(
    "data/market_context/private/rejected-opportunity-outcomes-v3.jsonl"
)
OUTPUT_ROOT = Path("output/research/active_swing/rejected_shadow")
MINIMUM_GATE_SAMPLE = 30
REQUIRED_GATE_CATEGORIES = (
    "MTF",
    "HTF",
    "TECHNICAL_SCORE",
    "STRUCTURE",
    "VOLUME",
    "RELATIVE_STRENGTH",
    "SECTOR",
    "INDUSTRY",
    "FUNDAMENTALS",
    "EARNINGS",
    "NEWS",
    "SEC",
    "MACRO",
    "OPTIONS",
    "SPREAD",
    "LIQUIDITY",
    "EXPECTED_VALUE",
    "ECR",
    "PORTFOLIO_HEAT",
    "CORRELATION",
    "SHARIAH",
    "STALE_DATA",
)
AUTHORITY = {
    "strategy_authority": "NONE",
    "execution_authority": "NONE",
    "automatic_orders": 0,
    "broker_calls": 0,
    "order_calls": 0,
}


def settle_rejected_opportunities(
    project_root: Path,
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate rejected setups without changing their canonical outcome."""
    now = _utc(observed_at or datetime.now(UTC))
    episode_path = project_root / PRIVATE_EPISODES
    outcome_path = project_root / PRIVATE_OUTCOMES
    canonical_outcomes = (
        project_root
        / "data/market_context/private/entry-episode-outcomes.jsonl"
    )
    canonical_hash_before = _file_hash(canonical_outcomes)
    episodes = [
        row
        for row in _read_jsonl(episode_path)
        if row.get("schema") == "active_swing_forward_episode_v1"
        and row.get("feature_snapshot_hash")
        and _rejection_codes(row)
    ]
    existing = {
        str(row.get("episode_id")): row
        for row in _read_jsonl(outcome_path)
        if row.get("episode_id")
    }
    pending = [
        row
        for row in episodes
        if str(row.get("episode_id")) not in existing
    ]
    frames = (
        _load_current_frames(project_root, observed_at=now)
        if pending
        else {}
    )
    evaluated: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    for episode in pending:
        timeframe = str(episode.get("timeframe") or "")
        symbol = str(episode.get("symbol") or "").upper()
        counterfactual_episode = dict(episode)
        counterfactual_episode["state"] = "COUNTERFACTUAL_SHADOW_EVALUATION"
        result = _evaluate_episode(
            counterfactual_episode,
            frame=frames.get(timeframe, {}).get(symbol),
            evaluated_at=now,
            counterfactual_long=True,
        )
        wrapped = _counterfactual_result(
            episode,
            result,
            symbol=symbol,
        )
        evaluated.append(wrapped)
        if wrapped["terminal"]:
            terminal.append(wrapped)
    _append_terminal(outcome_path, terminal)
    stored = list(existing.values()) + terminal
    attribution = build_gate_value_attribution(stored, generated_at=now)
    _write_json(project_root / OUTPUT_ROOT / "gate-attribution.json", attribution)
    pd.DataFrame(attribution["gates"]).to_parquet(
        project_root / OUTPUT_ROOT / "gate-attribution.parquet",
        index=False,
    )
    canonical_hash_after = _file_hash(canonical_outcomes)
    status = {
        "schema": "rejected_opportunity_shadow_status_v1",
        "status": "GO" if episodes else "NO_REJECTED_OPPORTUNITIES",
        "generated_at": now.isoformat(),
        "rejected_episode_count": len(episodes),
        "terminal_counterfactual_count": len(stored),
        "new_terminal_counterfactual_count": len(terminal),
        "pending_counterfactual_count": sum(
            not bool(row.get("terminal")) for row in evaluated
        ),
        "counterfactual_fill_count": sum(
            bool(row.get("would_fill")) for row in stored
        ),
        "counterfactual_performance_count": sum(
            _finite(row.get("net_R")) for row in stored
        ),
        "gate_count": len(attribution["gates"]),
        "gate_sample_ready_count": sum(
            row["sample_status"] == "EVALUABLE"
            for row in attribution["gates"]
        ),
        "canonical_outcome_store_hash_before": canonical_hash_before,
        "canonical_outcome_store_hash_after": canonical_hash_after,
        "canonical_outcome_store_unchanged": (
            canonical_hash_before == canonical_hash_after
        ),
        "private_counterfactual_store": str(outcome_path),
        "canonical_training_eligible": False,
        "counterfactuals_can_change_gates_automatically": False,
        "counterfactuals_can_change_authority": False,
        "raw_symbols_published": False,
        **AUTHORITY,
    }
    status["content_hash"] = stable_hash(status)
    _write_json(project_root / OUTPUT_ROOT / "status.json", status)
    return status


def build_gate_value_attribution(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcomes:
        if not row.get("terminal"):
            continue
        for gate in row.get("rejection_codes", []):
            grouped[str(gate)].append(row)
    rows: list[dict[str, Any]] = []
    for gate, values in sorted(grouped.items()):
        net_values = [
            float(row["net_R"])
            for row in values
            if _finite(row.get("net_R"))
        ]
        mean_net_r = _mean(net_values)
        wins = sum(value > 0 for value in net_values)
        losses = sum(value < 0 for value in net_values)
        zero_results = sum(value == 0 for value in net_values)
        gains = sum(value for value in net_values if value > 0)
        gross_losses = abs(sum(value for value in net_values if value < 0))
        profit_factor, profit_factor_status = _profit_factor(
            gains,
            gross_losses,
            sample_count=len(net_values),
        )
        posterior = _beta_posterior(wins=wins, losses=losses)
        confidence = _bootstrap_mean_interval(net_values, gate=gate)
        evidence_status = _evidence_status(
            confidence,
            sample_count=len(net_values),
        )
        rows.append(
            {
                "gate": gate,
                "gate_category": _gate_category(gate),
                "observations": len(values),
                "rejected_episode_count": len(values),
                "fills": sum(bool(row.get("would_fill")) for row in values),
                "counterfactual_fill_count": sum(
                    bool(row.get("would_fill")) for row in values
                ),
                "wins": wins,
                "losses": losses,
                "zero_results": zero_results,
                "unfilled": sum(
                    not bool(row.get("would_fill"))
                    and str(row.get("terminal_status"))
                    == "NO_FILL_EXPIRED"
                    for row in values
                ),
                "ambiguous": sum(
                    str(row.get("terminal_status"))
                    == "INTRABAR_PATH_AMBIGUOUS"
                    for row in values
                ),
                "performance_sample_count": len(net_values),
                "mean_counterfactual_net_R": mean_net_r,
                "median_counterfactual_net_R": _median(net_values),
                "profit_factor": profit_factor,
                "profit_factor_status": profit_factor_status,
                "hit_rate": _ratio(wins, wins + losses),
                "median_MFE_R": _median(
                    _finite_values(values, "maximum_favourable_excursion")
                ),
                "median_MAE_R": _median(
                    _finite_values(values, "maximum_adverse_excursion")
                ),
                "positive_counterfactual_ratio": _ratio(
                    wins,
                    len(net_values),
                ),
                "avoided_losses_R": round(gross_losses, 8),
                "missed_winners_R": round(gains, 8),
                "net_gate_contribution_R": round(
                    gross_losses - gains,
                    8,
                ),
                "estimated_protective_value_R": (
                    round(-mean_net_r, 8)
                    if mean_net_r is not None
                    else None
                ),
                "net_R_bootstrap_10pct": confidence[0],
                "net_R_bootstrap_90pct": confidence[1],
                "posterior_success_mean": posterior[0],
                "posterior_success_10pct": posterior[1],
                "posterior_success_90pct": posterior[2],
                "posterior_prior": "BETA_1_1",
                "sample_status": (
                    "EVALUABLE"
                    if len(net_values) >= MINIMUM_GATE_SAMPLE
                    else "INSUFFICIENT_SAMPLE"
                ),
                "evidence_status": evidence_status,
                "evidence_quality": _evidence_quality(
                    wins=wins,
                    losses=losses,
                    sample_count=len(net_values),
                ),
                "observational_assessment": _gate_assessment(
                    mean_net_r,
                    sample_count=len(net_values),
                ),
            }
        )
    report = {
        "schema": "rejected_opportunity_gate_value_attribution_v1",
        "status": "GO" if rows else "INSUFFICIENT_SAMPLE",
        "generated_at": _utc(generated_at or datetime.now(UTC)).isoformat(),
        "minimum_performance_sample_per_gate": MINIMUM_GATE_SAMPLE,
        "gates": rows,
        "required_gate_coverage": [
            {
                "gate_category": category,
                "observed_gate_count": sum(
                    row["gate_category"] == category for row in rows
                ),
                "status": (
                    "OBSERVED"
                    if any(row["gate_category"] == category for row in rows)
                    else "NO_COUNTERFACTUAL_OBSERVATIONS"
                ),
            }
            for category in REQUIRED_GATE_CATEGORIES
        ],
        "unobserved_required_gate_categories": [
            category
            for category in REQUIRED_GATE_CATEGORIES
            if not any(row["gate_category"] == category for row in rows)
        ],
        "attribution_method": "OVERLAPPING_OBSERVATIONAL_COUNTERFACTUALS",
        "causal_gate_value_claimed": False,
        "bayesian_binary_prior": "BETA_1_1",
        "net_R_uncertainty_method": "DETERMINISTIC_BOOTSTRAP_MEAN_10_90",
        "automatic_gate_relaxation": False,
        "canonical_training_eligible": False,
        "raw_symbols_published": False,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    return report


def rejected_shadow_status(project_root: Path) -> dict[str, Any]:
    path = project_root / OUTPUT_ROOT / "status.json"
    if not path.is_file():
        return {
            "schema": "rejected_opportunity_shadow_status_v1",
            "status": "NOT_RUN",
            **AUTHORITY,
        }
    return _read_json(path)


def gate_value_attribution_status(project_root: Path) -> dict[str, Any]:
    path = project_root / OUTPUT_ROOT / "gate-attribution.json"
    if not path.is_file():
        return {
            "schema": "rejected_opportunity_gate_value_attribution_v1",
            "status": "NOT_RUN",
            **AUTHORITY,
        }
    return _read_json(path)


def _counterfactual_result(
    episode: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    symbol: str,
) -> dict[str, Any]:
    row = {
        **result,
        "schema": "rejected_opportunity_counterfactual_outcome_v1",
        "symbol": symbol,
        "rejection_codes": _rejection_codes(episode),
        "original_state": episode.get("state"),
        "outcome_scope": "COUNTERFACTUAL_SHADOW_ONLY",
        "counterfactual_evaluator_version": "v3-explicit-long-envelope",
        "canonical_training_eligible": False,
        "execution_evidence": False,
        "automatic_gate_relaxation_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
        "order_calls": 0,
    }
    row["counterfactual_outcome_hash"] = stable_hash(row)
    return row


def _rejection_codes(episode: Mapping[str, Any]) -> list[str]:
    decision = episode.get("decision_contract", {})
    codes = {
        str(value)
        for key in ("hard_vetoes", "soft_vetoes")
        for value in decision.get(key, [])
        if value
    }
    state = str(episode.get("state") or "")
    if state.startswith("WATCHLIST_") and not codes:
        codes.add(state)
    return sorted(codes)


def _gate_assessment(
    mean_net_r: float | None,
    *,
    sample_count: int,
) -> str:
    if sample_count < MINIMUM_GATE_SAMPLE or mean_net_r is None:
        return "INSUFFICIENT_SAMPLE_NO_POLICY_CHANGE"
    if mean_net_r <= -0.05:
        return "OBSERVATIONALLY_PROTECTIVE_REQUIRES_ABLATION"
    if mean_net_r >= 0.05:
        return "POTENTIALLY_TOO_STRICT_REQUIRES_ABLATION"
    return "OBSERVATIONALLY_NEUTRAL_REQUIRES_ABLATION"


def _gate_category(gate: str) -> str:
    value = gate.upper()
    mappings = (
        ("STALE_DATA", ("STALE", "CURRENT_PRICE_REFERENCE")),
        ("MTF", ("MTF", "TIMEFRAME_HIERARCHY")),
        ("HTF", ("HTF", "HIGHER_TIMEFRAME")),
        ("RELATIVE_STRENGTH", ("RELATIVE_STRENGTH",)),
        ("PORTFOLIO_HEAT", ("PORTFOLIO_HEAT",)),
        ("EXPECTED_VALUE", ("EXPECTED_VALUE",)),
        ("FUNDAMENTALS", ("FUNDAMENTAL",)),
        ("CORRELATION", ("CORRELATION",)),
        ("LIQUIDITY", ("LIQUIDITY",)),
        ("STRUCTURE", ("STRUCTURE",)),
        ("TECHNICAL_SCORE", ("TECHNICAL", "ASSET_BIAS")),
        ("EARNINGS", ("EARNINGS",)),
        ("OPTIONS", ("OPTION", "GEX", "GAMMA")),
        ("SHARIAH", ("SHARIAH",)),
        ("SPREAD", ("SPREAD",)),
        ("VOLUME", ("VOLUME",)),
        ("SECTOR", ("SECTOR",)),
        ("INDUSTRY", ("INDUSTRY",)),
        ("MACRO", ("MACRO",)),
        ("NEWS", ("NEWS", "EVENT_RISK")),
        ("SEC", ("SEC_", "FILING")),
        ("ECR", ("ECR",)),
    )
    for category, markers in mappings:
        if any(marker in value for marker in markers):
            return category
    return "OTHER_EXPLICIT_GATE"


def _profit_factor(
    gains: float,
    losses: float,
    *,
    sample_count: int,
) -> tuple[float | None, str]:
    if sample_count == 0:
        return None, "NO_CLOSED_TRADES"
    if losses == 0 and gains > 0:
        return None, "PERFECT_NO_LOSSES"
    if losses == 0:
        return None, "ZERO_DENOMINATOR"
    return round(gains / losses, 8), "DEFINED"


def _beta_posterior(*, wins: int, losses: int) -> tuple[float, float, float]:
    alpha = 1 + wins
    beta = 1 + losses
    return (
        round(alpha / (alpha + beta), 8),
        round(float(beta_distribution.ppf(0.10, alpha, beta)), 8),
        round(float(beta_distribution.ppf(0.90, alpha, beta)), 8),
    )


def _bootstrap_mean_interval(
    values: list[float],
    *,
    gate: str,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        value = round(values[0], 8)
        return value, value
    seed = int(hashlib.sha256(gate.encode("utf-8")).hexdigest()[:8], 16)
    generator = pd.Series(range(2_000)).map(
        lambda index: _bootstrap_mean(values, seed=seed + int(index))
    )
    return (
        round(float(generator.quantile(0.10)), 8),
        round(float(generator.quantile(0.90)), 8),
    )


def _bootstrap_mean(values: list[float], *, seed: int) -> float:
    sample = pd.Series(values).sample(
        n=len(values),
        replace=True,
        random_state=seed,
    )
    return float(sample.mean())


def _evidence_status(
    confidence: tuple[float | None, float | None],
    *,
    sample_count: int,
) -> str:
    lower, upper = confidence
    if sample_count < MINIMUM_GATE_SAMPLE or lower is None or upper is None:
        return "INSUFFICIENT_EVIDENCE"
    if upper < 0:
        return "LIKELY_VALUE_ADD"
    if lower > 0:
        return "LIKELY_TOO_STRICT"
    if lower <= 0 <= upper:
        return "NEUTRAL"
    return "HARMFUL"


def _evidence_quality(*, wins: int, losses: int, sample_count: int) -> str:
    if sample_count >= MINIMUM_GATE_SAMPLE:
        return "EVALUABLE"
    if sample_count >= 3 and losses > wins:
        return "EARLY_SUPPORTIVE_EVIDENCE"
    if sample_count >= 3 and wins > losses:
        return "EARLY_QUESTIONABLE_EVIDENCE"
    return "INSUFFICIENT_EVIDENCE"


def _finite_values(
    rows: Iterable[Mapping[str, Any]],
    key: str,
) -> list[float]:
    return [float(row[key]) for row in rows if _finite(row.get(key))]


def _append_terminal(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=10):
        existing = {
            str(row.get("episode_id"))
            for row in _read_jsonl(path)
            if row.get("episode_id")
        }
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                episode_id = str(row.get("episode_id"))
                if not episode_id or episode_id in existing:
                    continue
                handle.write(
                    json.dumps(row, sort_keys=True, default=str) + "\n"
                )
                existing.add(episode_id)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 8) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(pd.Series(values).median()), 8)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 8) if denominator else None


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "build_gate_value_attribution",
    "gate_value_attribution_status",
    "rejected_shadow_status",
    "settle_rejected_opportunities",
]
