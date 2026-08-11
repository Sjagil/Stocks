from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.execution.idempotency import stable_hash


ROLE_SPECS = {
    "strategic_allocation": {
        "timeframes": {"1w", "1mo"},
        "purpose": "asset, sector and sleeve allocation",
        "holding_period": "weeks to months",
    },
    "active_swing": {
        "timeframes": {"1h", "2h", "4h", "1d"},
        "purpose": "active setups held from hours to several weeks",
        "holding_period": "hours to weeks",
    },
}


def publish_role_leaderboards(project_root: Path) -> dict[str, Any]:
    source = (
        project_root
        / "output/research/phase11_12/strategy-summary.parquet"
    )
    summary = pd.read_parquet(source) if source.is_file() else pd.DataFrame()
    generated_at = datetime.now(UTC).isoformat()
    reports: dict[str, dict[str, Any]] = {}
    for role, specification in ROLE_SPECS.items():
        frame = _role_frame(summary, role=role, specification=specification)
        reports[role] = _publish_role(
            project_root,
            role=role,
            frame=frame,
            generated_at=generated_at,
            specification=specification,
        )
    tactical = _tactical_frame(project_root)
    reports["tactical_execution"] = _publish_role(
        project_root,
        role="tactical_execution",
        frame=tactical,
        generated_at=generated_at,
        specification={
            "timeframes": {"1h", "2h", "4h"},
            "purpose": "entry timing overlay, never standalone alpha",
            "holding_period": "entry observation only",
        },
    )
    exploratory = _exploratory_frame(project_root)
    reports["exploratory_forward"] = _publish_role(
        project_root,
        role="exploratory_forward",
        frame=exploratory,
        generated_at=generated_at,
        specification={
            "timeframes": {"1h", "2h", "4h", "1d", "1w", "1mo"},
            "purpose": "append-only forward observation without promotion",
            "holding_period": "strategy defined",
        },
    )
    payload = {
        "schema": "functional_research_leaderboards_v1",
        "status": "GO" if any(row["row_count"] for row in reports.values()) else "DATA_UNAVAILABLE",
        "generated_at": generated_at,
        "roles": reports,
        "global_cross_role_ranking_allowed": False,
        "selection_bias_status": "BLOCKED_NO_NEW_INDEPENDENT_HOLDOUT",
        "financial_finalist": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
    }
    payload["content_hash"] = stable_hash(payload)
    _write_json(
        project_root / "output/research/role_leaderboards/status.json",
        payload,
    )
    return payload


def role_leaderboards_status(project_root: Path) -> dict[str, Any]:
    path = project_root / "output/research/role_leaderboards/status.json"
    if not path.is_file():
        return {
            "schema": "functional_research_leaderboards_v1",
            "status": "NOT_PUBLISHED",
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "order_calls": 0,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _role_frame(
    summary: pd.DataFrame,
    *,
    role: str,
    specification: dict[str, Any],
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    work = summary.copy()
    numeric = (
        "CAGR",
        "Sharpe",
        "period_profit_factor",
        "stress_50bps_profit_factor",
        "maximum_drawdown",
        "fill_count",
    )
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    gate = (
        work["status"].eq("COMPLETE")
        & work["timeframe"].isin(specification["timeframes"])
        & work["CAGR"].gt(0)
        & work["period_profit_factor"].gt(1.0)
        & work["stress_50bps_profit_factor"].gt(1.0)
        & work["maximum_drawdown"].ge(-0.35)
        & work["fill_count"].ge(20 if role == "strategic_allocation" else 50)
    )
    selected = work.loc[gate].copy()
    if selected.empty:
        return selected
    selected["role"] = role.upper()
    selected["rank_score"] = _rank_score(selected)
    selected["subrole"] = [
        _subrole(role, str(value)) for value in selected["timeframe"]
    ]
    selected = selected.sort_values(
        [
            "rank_score",
            "Sharpe",
            "stress_50bps_profit_factor",
            "period_profit_factor",
        ],
        ascending=False,
    )
    if "economic_outcome_fingerprint" in selected:
        selected = selected.drop_duplicates(
            "economic_outcome_fingerprint", keep="first"
        )
    return selected.drop_duplicates(
        ["formula", "timeframe", "asset_class"], keep="first"
    ).head(100)


def _tactical_frame(project_root: Path) -> pd.DataFrame:
    path = project_root / "output/research/phase11_15/architecture-summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["role"] = "TACTICAL_EXECUTION"
    frame["subrole"] = "ENTRY_OVERLAY"
    frame["formula"] = (
        frame["entry_strategy"].astype(str)
        + "::"
        + frame["architecture"].astype(str)
    )
    frame["timeframe"] = frame["lower_timeframe"]
    frame["rank_score"] = (
        pd.to_numeric(frame["median_incremental_CAGR"], errors="coerce")
        .rank(pct=True)
        .fillna(0)
        * 0.45
        + pd.to_numeric(frame["median_incremental_pf"], errors="coerce")
        .rank(pct=True)
        .fillna(0)
        * 0.35
        + pd.to_numeric(frame["flow_improvement_ratio"], errors="coerce")
        .fillna(0)
        * 0.20
    )
    frame["financial_gate"] = frame[
        "incremental_flow_evidence_status"
    ].astype(str)
    frame["data_class"] = "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW"
    frame["standalone_strategy"] = False
    return frame.sort_values("rank_score", ascending=False).head(100)


def _exploratory_frame(project_root: Path) -> pd.DataFrame:
    path = (
        project_root
        / "output/research/phase11_12/forward-performance.json"
    )
    if not path.is_file():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("strategies") or payload.get("observations") or []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["role"] = "EXPLORATORY_FORWARD"
    frame["rank_score"] = 0.0
    frame["automatic_promotion"] = False
    return frame.head(100)


def _publish_role(
    project_root: Path,
    *,
    role: str,
    frame: pd.DataFrame,
    generated_at: str,
    specification: dict[str, Any],
) -> dict[str, Any]:
    root = project_root / "output/research" / role
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "leaderboard.parquet", index=False)
    rows = frame.replace({np.nan: None}).to_dict("records")
    status = (
        "GO"
        if role != "tactical_execution" and rows
        else "NO_INCREMENTAL_OBSERVED_FLOW_EVIDENCE"
        if role == "tactical_execution"
        and not any(
            str(row.get("financial_gate")) == "POSITIVE" for row in rows
        )
        else "GO"
        if rows
        else "DATA_UNAVAILABLE"
    )
    payload = {
        "schema": "functional_research_leaderboard_v1",
        "status": status,
        "role": role.upper(),
        "purpose": specification["purpose"],
        "holding_period": specification["holding_period"],
        "timeframes": sorted(specification["timeframes"]),
        "row_count": len(rows),
        "generated_at": generated_at,
        "leaderboard": rows,
        "cross_role_comparison_allowed": False,
        "financial_finalist": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
    }
    payload["content_hash"] = stable_hash(payload)
    _write_json(root / "leaderboard.json", payload)
    _write_json(root / "status.json", {key: value for key, value in payload.items() if key != "leaderboard"})
    return {
        "status": status,
        "row_count": len(rows),
        "timeframes": sorted(specification["timeframes"]),
        "path": str(root / "leaderboard.json"),
    }


def _rank_score(frame: pd.DataFrame) -> pd.Series:
    drawdown_quality = 1.0 - frame["maximum_drawdown"].abs().rank(pct=True)
    fill_quality = np.log1p(frame["fill_count"]).rank(pct=True)
    return (
        0.30 * frame["Sharpe"].rank(pct=True)
        + 0.20 * frame["stress_50bps_profit_factor"].rank(pct=True)
        + 0.15 * frame["period_profit_factor"].rank(pct=True)
        + 0.15 * frame["CAGR"].rank(pct=True)
        + 0.10 * drawdown_quality
        + 0.10 * fill_quality
    )


def _subrole(role: str, timeframe: str) -> str:
    if role == "strategic_allocation":
        return "STRUCTURAL_ALLOCATION"
    return (
        "SWING_SETUP"
        if timeframe in {"4h", "1d"}
        else "ACTIVE_ENTRY_REFINEMENT"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = ["publish_role_leaderboards", "role_leaderboards_status"]
