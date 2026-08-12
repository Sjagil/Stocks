from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.execution.idempotency import stable_hash


OUTPUT_PATH = Path("output/portfolio/p1-monitoring.json")

FREQUENCY_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "frequency": "15m",
        "role": "EARLIEST_TACTICAL_SWING_DECISION_AND_MANAGEMENT_LAYER",
        "features": [
            "price_change", "relative_volume", "spread", "gap_behavior",
            "breakout_pullback", "position_deterioration",
        ],
        "required_for_opportunity": "STRATEGY_CONTRACT_DEPENDENT",
    },
    {
        "frequency": "1h",
        "role": "PRIMARY_SWING_SETUP_DEVELOPMENT",
        "features": [
            "trend", "momentum", "breakout", "mean_reversion",
            "relative_strength", "volume", "volatility", "ranking",
        ],
        "required_for_opportunity": False,
    },
    {
        "frequency": "2h",
        "role": "PRIMARY_SWING_SETUP_DEVELOPMENT",
        "features": [
            "trend", "momentum", "pullback", "breakout", "continuation",
        ],
        "required_for_opportunity": "STRATEGY_CONTRACT_DEPENDENT",
    },
    {
        "frequency": "4h",
        "role": "SWING_STRUCTURE_AND_POSITION_HEALTH",
        "features": [
            "trend_quality", "volatility_regime", "support_resistance",
            "position_health",
        ],
        "required_for_opportunity": False,
    },
    {
        "frequency": "1d",
        "role": "CROSS_SECTIONAL_LEADERSHIP_AND_REGIME",
        "features": [
            "market_regime", "breadth", "sector_rotation",
            "commodity_trends", "ETF_leadership", "macro_sensitivity",
        ],
        "required_for_opportunity": True,
    },
    {
        "frequency": "6h",
        "role": "OPTIONAL_INTERMEDIATE_STRUCTURE_CONTEXT",
        "features": ["trend_quality", "regime_confirmation"],
        "required_for_opportunity": False,
    },
    {
        "frequency": "12h",
        "role": "OPTIONAL_INTERMEDIATE_STRUCTURE_CONTEXT",
        "features": ["trend_quality", "regime_confirmation"],
        "required_for_opportunity": False,
    },
    {
        "frequency": "1w",
        "role": "STRUCTURAL_TREND_AND_MAJOR_ROTATION_CONTEXT",
        "features": ["structural_trend", "relative_strength_rotation"],
        "required_for_opportunity": False,
    },
)


def publish_monitoring_architecture(project_root: Path) -> dict[str, Any]:
    coverage_path = project_root / "output/portfolio/coverage-waterfall.parquet"
    normalized_path = project_root / "output/portfolio/normalized-opportunities.json"
    coverage = (
        pd.read_parquet(coverage_path)
        if coverage_path.is_file()
        else pd.DataFrame()
    )
    normalized = _read_json(normalized_path)
    opportunities = normalized.get("combined_ranking", [])
    frequency_state = [
        {**contract, **_frequency_availability(project_root, contract["frequency"])}
        for contract in FREQUENCY_CONTRACTS
    ]
    asset_classes: dict[str, Any] = {}
    for asset_class in ("EQUITY", "ETF", "COMMODITY_EXPOSURE"):
        selected = (
            coverage.loc[coverage["asset_class"] == asset_class]
            if not coverage.empty
            else coverage
        )
        candidate_rows = [
            row for row in opportunities if row.get("asset_class") == asset_class
        ]
        asset_classes[asset_class] = {
            "discovered": len(selected),
            "data_analyzable": (
                int(selected["data_analyzable"].sum())
                if not selected.empty
                else 0
            ),
            "research_opportunities": len(candidate_rows),
            "fresh_opportunities": sum(
                _fresh(row.get("signal_timestamp") or row.get("data_timestamp"))
                for row in candidate_rows
            ),
            "monitored_frequencies": [row["frequency"] for row in frequency_state],
            "first_class_source": True,
        }
    current = _read_json(
        project_root / "output/portfolio/current_allocation.json"
    )
    targets = _read_json(
        project_root / "output/portfolio/desired-portfolio-targets.json"
    )
    report: dict[str, Any] = {
        "schema": "multi_asset_swing_monitoring_architecture_v1",
        "status": (
            "GO"
            if all(row["data_analyzable"] > 0 for row in asset_classes.values())
            else "NO_GO"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "frequencies": frequency_state,
        "asset_classes": asset_classes,
        "held_asset_count": int(current.get("position_count") or 0),
        "unheld_opportunity_count": len(opportunities),
        "held_and_unheld_use_same_intelligence": True,
        "position_rescoring_contract": [
            "THESIS_VALID", "THESIS_WEAKENING", "STOP_THREATENED",
            "UNDERPERFORMING", "CAPITAL_INEFFICIENT", "EVENT_RISK_HIGH",
            "ROTATION_CANDIDATE", "EXIT_REQUIRED",
        ],
        "rotation_decision_count": len(targets.get("rotation_decisions", [])),
        "all_timeframes_need_not_agree": True,
        "higher_timeframes_are_context_not_automatic_veto": True,
        "high_frequency_trading": False,
        "critical_stale_data_invalidates_actionability": True,
        "optional_stale_context_reduces_confidence": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _frequency_availability(project_root: Path, frequency: str) -> dict[str, Any]:
    root = project_root / "data/research/multitimeframe/private"
    pattern = f"interval={frequency}"
    paths = [path for path in root.rglob("bars.parquet") if pattern in path.parts]
    symbols = {
        part.split("=", 1)[1].upper()
        for path in paths
        for part in path.parts
        if part.startswith("symbol=")
    }
    if frequency == "1d":
        symbols.update(
            path.stem.upper()
            for path in (
                project_root / "data/research/critical_trading/yfinance"
            ).glob("*.parquet")
        )
    return {
        "dataset_count": len(paths),
        "symbol_count": len(symbols),
        "availability": "AVAILABLE" if symbols else "OPTIONAL_DATA_UNAVAILABLE",
        "scheduler_contract": "CONTINUOUS_CADENCE_EXTERNAL_ORCHESTRATOR",
    }


def _fresh(value: Any) -> bool:
    try:
        timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return False
    if pd.isna(timestamp):
        return False
    return (pd.Timestamp.now(tz="UTC") - timestamp).days <= 5


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


__all__ = ["FREQUENCY_CONTRACTS", "publish_monitoring_architecture"]
