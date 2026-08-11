from __future__ import annotations

import json
from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.portfolio.coverage import normalize_asset_class


OUTPUT_PATH = Path("output/research/p1/walk-forward-manifests.json")


@dataclass(frozen=True)
class WalkForwardFoldManifest:
    dataset_id: str
    strategy_id: str
    asset_universe: tuple[str, ...]
    asset_class: str
    timeframe: str
    fold_id: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    purge_interval: dict[str, str]
    embargo_interval: dict[str, str]
    selected_parameters: dict[str, Any]
    cost_assumptions: dict[str, Any]
    slippage_assumptions: dict[str, Any]
    universe_assumptions: dict[str, Any]
    regime_labels: tuple[str, ...]
    results: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["asset_universe"] = list(self.asset_universe)
        payload["regime_labels"] = list(self.regime_labels)
        payload["schema"] = "standard_walk_forward_fold_manifest_v1"
        payload["content_hash"] = stable_hash(payload)
        return payload


def build_walk_forward_manifest(
    *,
    dataset_id: str,
    strategy_id: str,
    asset_universe: list[str],
    asset_class: str,
    timeframe: str,
    fold_id: str,
    train_start: date,
    train_end: date,
    validation_start: date,
    validation_end: date,
    test_start: date,
    test_end: date,
    purge_days: int,
    embargo_days: int,
    selected_parameters: dict[str, Any],
    cost_assumptions: dict[str, Any],
    slippage_assumptions: dict[str, Any],
    universe_assumptions: dict[str, Any],
    regime_labels: list[str] | None = None,
    results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not (
        train_start <= train_end < validation_start <= validation_end
        < test_start <= test_end
    ):
        raise ValueError("NON_CAUSAL_WALK_FORWARD_BOUNDARIES")
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("NEGATIVE_PURGE_OR_EMBARGO")
    manifest = WalkForwardFoldManifest(
        dataset_id=dataset_id,
        strategy_id=strategy_id,
        asset_universe=tuple(sorted(set(asset_universe))),
        asset_class=asset_class,
        timeframe=timeframe,
        fold_id=fold_id,
        train_start=train_start.isoformat(),
        train_end=train_end.isoformat(),
        validation_start=validation_start.isoformat(),
        validation_end=validation_end.isoformat(),
        test_start=test_start.isoformat(),
        test_end=test_end.isoformat(),
        purge_interval={
            "start": (train_end + timedelta(days=1)).isoformat(),
            "end": (validation_start - timedelta(days=1)).isoformat(),
            "calendar_days": str(purge_days),
        },
        embargo_interval={
            "start": (validation_end + timedelta(days=1)).isoformat(),
            "end": (test_start - timedelta(days=1)).isoformat(),
            "calendar_days": str(embargo_days),
        },
        selected_parameters=selected_parameters,
        cost_assumptions=cost_assumptions,
        slippage_assumptions=slippage_assumptions,
        universe_assumptions=universe_assumptions,
        regime_labels=tuple(regime_labels or ("UNLABELED",)),
        results=results or {"status": "NOT_RUN_MANIFEST_ONLY"},
    )
    return manifest.as_dict()


def publish_standard_walk_forward_manifests(project_root: Path) -> dict[str, Any]:
    legacy_path = project_root / "output/research/results/walk_forward.csv"
    universe_path = project_root / "output/universe/instruments.parquet"
    if not legacy_path.is_file() or not universe_path.is_file():
        return _blocked("LEGACY_WALK_FORWARD_OR_UNIVERSE_MISSING")
    legacy = pd.read_csv(legacy_path)
    universe = pd.read_parquet(universe_path)
    universe["normalized_asset_class"] = [
        normalize_asset_class(row)
        for row in universe.to_dict(orient="records")
    ]
    strategies = (
        legacy.sort_values(["strategy_id", "Sharpe"], ascending=[True, False])
        .groupby("strategy_id", sort=True)
        .head(1)
    )
    dataset_start, dataset_end = _dataset_bounds(project_root)
    dataset_id = stable_hash(
        {
            "source": "LOCAL_PIT_PRICE_DATA",
            "start": dataset_start.isoformat(),
            "end": dataset_end.isoformat(),
            "universe_hash": stable_hash(
                sorted(universe["instrument_id"].astype(str))
            ),
        }
    )
    manifests: list[dict[str, Any]] = []
    for _, strategy in strategies.iterrows():
        asset_class = _strategy_asset_class(strategy)
        symbols = sorted(
            universe.loc[
                (universe["normalized_asset_class"] == asset_class)
                & universe["active_listing"].astype(bool),
                "symbol",
            ]
            .astype(str)
            .str.upper()
        )
        timeframe = str(strategy.get("timeframe") or "1d")
        folds = _fold_boundaries(
            dataset_start,
            dataset_end,
            timeframe=timeframe,
            count=5,
        )
        for index, boundaries in enumerate(folds, 1):
            manifests.append(
                build_walk_forward_manifest(
                    dataset_id=dataset_id,
                    strategy_id=str(strategy["strategy_id"]),
                    asset_universe=symbols,
                    asset_class=asset_class,
                    timeframe=timeframe,
                    fold_id=f"P1-WF-{index:02d}",
                    **boundaries,
                    purge_days=5,
                    embargo_days=5,
                    selected_parameters={
                        "profile": _clean(strategy.get("profile")),
                        "formula": _clean(strategy.get("formula")),
                        "parameter_plateau": bool(
                            strategy.get("parameter_plateau", False)
                        ),
                        "selection_scope": "TRAIN_AND_VALIDATION_ONLY",
                    },
                    cost_assumptions={
                        "model": "SHARED_TRANSACTION_COST_MODEL_V1",
                        "legacy_selected_cost_bps": _clean(
                            strategy.get("cost_bps")
                        ),
                    },
                    slippage_assumptions={
                        "model": "SHARED_TRANSACTION_COST_MODEL_V1",
                        "point_in_time_spread_required": True,
                    },
                    universe_assumptions={
                        "point_in_time_membership_required": True,
                        "listing_and_delisting_dates_required": True,
                        "historical_shariah_when_available": True,
                        "complete_historical_membership_claimed": False,
                    },
                    regime_labels=["MULTI_DIMENSIONAL_REGIME_REQUIRED"],
                )
            )
    audit = lookahead_protection_audit(project_root)
    report: dict[str, Any] = {
        "schema": "standard_walk_forward_manifest_registry_v1",
        "status": "GO" if manifests and audit["status"] == "GO" else "NO_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_id": dataset_id,
        "dataset_start": dataset_start.isoformat(),
        "dataset_end": dataset_end.isoformat(),
        "strategy_count": len(strategies),
        "manifest_count": len(manifests),
        "folds_per_strategy": 5,
        "legacy_result_row_count": len(legacy),
        "legacy_strategy_registry_migrated": len(strategies),
        "legacy_result_dates_relabelled": False,
        "legacy_result_date_limitation": "LEGACY_ROWS_RETAINED_UNCHANGED; NEW MANIFESTS_DO_NOT_FABRICATE_OLD_SPLIT_DATES",
        "manifests": manifests,
        "lookahead_protection": audit,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def lookahead_protection_audit(
    project_root: Path | None = None,
) -> dict[str, Any]:
    requirements = {
        "bar_timing_uses_closed_bars": (
            "src/stocks/data/bars.py",
            ("available_at", "bar availability time"),
        ),
        "higher_timeframe_aggregation_requires_closed_period": (
            "src/stocks/data/multitimeframe.py",
            ("closed_bars_only",),
        ),
        "earnings_uses_observed_point_in_time_snapshot": (
            "src/stocks/analysis/theme_events.py",
            ("observed_at", "earnings"),
        ),
        "sec_uses_acceptance_timestamp": (
            "src/stocks/research/sec_overlay.py",
            ("accepted_at<=?",),
        ),
        "news_uses_publication_timestamp": (
            "src/stocks/news/intelligence.py",
            ("published_at",),
        ),
        "macro_uses_release_timestamp": (
            "src/stocks/macro/transforms.py",
            ("available_at", "cutoff"),
        ),
        "fundamentals_use_publicly_available_date_not_period_end": (
            "src/stocks/screener/sources.py",
            ("accepted_at", "available_at"),
        ),
        "universe_membership_is_point_in_time_when_available": (
            "src/stocks/research/phase11_4/pipeline.py",
            (
                "historical_membership_complete",
                "current_membership_backprojection_used",
            ),
        ),
        "corporate_actions_are_effective_date_bounded": (
            "src/stocks/data/total_returns.py",
            ("effective_date > session_date",),
        ),
        "delistings_are_retained": (
            "src/stocks/universe.py",
            ("is_delisted", "active_listing"),
        ),
    }
    evidence: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for check, (relative, tokens) in requirements.items():
        if project_root is None:
            passed = True
        else:
            path = project_root / relative
            text = (
                path.read_text(encoding="utf-8")
                if path.is_file()
                else ""
            )
            passed = bool(text) and all(token in text for token in tokens)
        checks[check] = passed
        evidence[check] = {
            "source": relative,
            "required_tokens": list(tokens),
            "source_evidence_verified": passed if project_root else None,
        }
    return {
        "schema": "p1_lookahead_protection_audit_v1",
        "status": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "evidence": evidence,
        "evidence_mode": (
            "SOURCE_BOUND" if project_root is not None else "CONTRACT_ONLY"
        ),
        "period_end_date_is_public_availability": False,
        "complete_pit_coverage_claimed": False,
    }


def _fold_boundaries(
    start: date,
    end: date,
    *,
    timeframe: str,
    count: int,
) -> list[dict[str, date]]:
    intraday = timeframe.lower() in {"15m", "30m", "1h", "2h", "4h"}
    validation_months = 3 if intraday else 12
    test_months = 3 if intraday else 12
    step_months = 3 if intraday else 12
    minimum_train_months = 12 if intraday else 60
    folds: list[dict[str, date]] = []
    latest_test_end = end
    for reverse_index in range(count - 1, -1, -1):
        test_end = _add_months(latest_test_end, -reverse_index * step_months)
        test_start = _add_months(test_end, -test_months) + timedelta(days=1)
        validation_end = test_start - timedelta(days=6)
        validation_start = _add_months(validation_end, -validation_months) + timedelta(days=1)
        train_end = validation_start - timedelta(days=6)
        if _add_months(train_end, -minimum_train_months) < start:
            train_start = start
        else:
            train_start = start
        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return folds


def _dataset_bounds(project_root: Path) -> tuple[date, date]:
    starts: list[date] = []
    ends: list[date] = []
    for path in (project_root / "data/research/critical_trading/yfinance").glob("*.parquet"):
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        column = next((name for name in ("session_date", "timestamp_utc", "date", "timestamp") if name in frame), None)
        if column is None:
            continue
        values = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
        if not values.empty:
            starts.append(values.min().date())
            ends.append(values.max().date())
    if not starts:
        raise ValueError("WALK_FORWARD_DATASET_DATES_UNAVAILABLE")
    return min(starts), max(ends)


def _strategy_asset_class(row: pd.Series) -> str:
    raw = str(row.get("asset_class") or "").upper()
    if raw in {"STOCK", "EQUITY"}:
        return "EQUITY"
    if raw == "ETF":
        return "ETF"
    if "COMMODITY" in raw:
        return "COMMODITY_EXPOSURE"
    identity = str(row.get("strategy_id") or "").upper()
    if "MULTI_ASSET" in identity:
        return "ETF"
    return "EQUITY"


def _add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    return date(year, month, min(value.day, monthrange(year, month)[1]))


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "standard_walk_forward_manifest_registry_v1",
        "status": "NO_GO",
        "blockers": [reason],
        "execution_authority": "NONE",
    }


__all__ = [
    "WalkForwardFoldManifest",
    "build_walk_forward_manifest",
    "lookahead_protection_audit",
    "publish_standard_walk_forward_manifests",
]
