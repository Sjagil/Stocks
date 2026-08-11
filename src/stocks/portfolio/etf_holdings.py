from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stocks.execution.idempotency import stable_hash


DATA_PATH = Path("data/research/p1/etf-holdings.parquet")
MANIFEST_PATH = Path("output/research/p1/etf-holdings-manifest.json")


def collect_etf_holdings(
    project_root: Path,
    *,
    symbols: Iterable[str],
) -> dict[str, Any]:
    import yfinance as yf

    requested = sorted(
        {str(symbol).upper() for symbol in symbols if str(symbol).strip()}
    )
    observed_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for symbol in requested:
        try:
            holdings = yf.Ticker(symbol).funds_data.top_holdings
        except Exception as exc:  # pragma: no cover - provider behavior
            statuses.append(
                {
                    "symbol": symbol,
                    "status": "PROVIDER_ERROR",
                    "reason": type(exc).__name__,
                    "holding_count": 0,
                }
            )
            continue
        if not isinstance(holdings, pd.DataFrame) or holdings.empty:
            statuses.append(
                {
                    "symbol": symbol,
                    "status": "NO_HOLDINGS_FOR_STRUCTURE",
                    "holding_count": 0,
                }
            )
            continue
        accepted = 0
        for holding_symbol, raw in holdings.iterrows():
            weight = _number(raw.get("Holding Percent"))
            if weight is None or weight <= 0:
                continue
            rows.append(
                {
                    "etf_symbol": symbol,
                    "holding_symbol": str(holding_symbol).upper(),
                    "holding_name": str(raw.get("Name") or holding_symbol),
                    "weight": weight,
                    "observed_at": observed_at,
                    "provider": "YFINANCE_FUNDS_DATA",
                }
            )
            accepted += 1
        statuses.append(
            {
                "symbol": symbol,
                "status": "GO" if accepted else "NO_VALID_HOLDINGS",
                "holding_count": accepted,
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "etf_symbol", "holding_symbol", "holding_name", "weight",
            "observed_at", "provider",
        ],
    )
    data_path = project_root / DATA_PATH
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data_path, index=False)
    manifest: dict[str, Any] = {
        "schema": "p1_etf_holdings_cache_manifest_v1",
        "status": "GO" if not frame.empty else "NO_DATA",
        "generated_at": observed_at,
        "requested_count": len(requested),
        "fund_count_with_holdings": int(frame["etf_symbol"].nunique()),
        "holding_row_count": len(frame),
        "statuses": statuses,
        "data_path": DATA_PATH.as_posix(),
        "top_holdings_only": True,
        "complete_portfolio_look_through_claimed": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    manifest["content_hash"] = stable_hash(manifest)
    path = project_root / MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_etf_holdings(project_root: Path) -> pd.DataFrame:
    path = project_root / DATA_PATH
    if not path.is_file():
        return pd.DataFrame(
            columns=["etf_symbol", "holding_symbol", "weight"]
        )
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame(
            columns=["etf_symbol", "holding_symbol", "weight"]
        )
    required = {"etf_symbol", "holding_symbol", "weight"}
    return frame if required.issubset(frame.columns) else pd.DataFrame()


def holdings_overlap_report(
    opportunities: Iterable[dict[str, Any]],
    holdings: pd.DataFrame,
    *,
    threshold: float = 0.25,
) -> dict[str, Any]:
    if holdings.empty:
        return {
            "status": "DATA_MISSING_FALLBACK_TO_SECTOR_AND_ECONOMIC_CLUSTER",
            "fund_count": 0,
            "fund_pairs": [],
            "direct_stock_overlaps": [],
            "top_holdings_only": True,
        }
    vectors = {
        str(symbol): {
            str(row["holding_symbol"]): float(row["weight"])
            for row in group.to_dict(orient="records")
        }
        for symbol, group in holdings.groupby("etf_symbol")
    }
    pairs = []
    names = sorted(vectors)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = _weighted_overlap(vectors[left], vectors[right])
            if overlap >= threshold:
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "weighted_top_holdings_overlap": round(overlap, 8),
                    }
                )
    opportunity_symbols = {
        str(row.get("symbol") or "").upper()
        for row in opportunities
        if row.get("asset_class") == "EQUITY"
    }
    direct = [
        {
            "etf": etf,
            "equity": holding,
            "holding_weight": round(weight, 8),
        }
        for etf, vector in sorted(vectors.items())
        for holding, weight in sorted(vector.items())
        if holding in opportunity_symbols
    ]
    return {
        "status": "TOP_HOLDINGS_LOOK_THROUGH_AVAILABLE",
        "fund_count": len(vectors),
        "fund_pairs": sorted(
            pairs,
            key=lambda row: (
                -float(row["weighted_top_holdings_overlap"]),
                row["left"], row["right"],
            ),
        ),
        "direct_stock_overlaps": sorted(
            direct,
            key=lambda row: (-float(row["holding_weight"]), row["etf"]),
        ),
        "overlap_threshold": threshold,
        "top_holdings_only": True,
        "complete_look_through_claimed": False,
    }


def _weighted_overlap(
    left: dict[str, float], right: dict[str, float]
) -> float:
    names = set(left) | set(right)
    denominator = sum(max(left.get(name, 0.0), right.get(name, 0.0)) for name in names)
    numerator = sum(min(left.get(name, 0.0), right.get(name, 0.0)) for name in names)
    return numerator / denominator if denominator > 0 else 0.0


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


__all__ = [
    "collect_etf_holdings",
    "holdings_overlap_report",
    "load_etf_holdings",
]
