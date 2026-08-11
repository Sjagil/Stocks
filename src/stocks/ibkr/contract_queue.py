from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.research.autopilot.contracts import stable_hash
from stocks.universe import broad_asset_metadata


FUNNEL_PATH = Path("output/portfolio/opportunity-funnel.json")
SCREENER_PATH = Path("output/screener/latest-preview.json")
ACTIVE_PLAN_PATH = Path("output/portfolio/active_portfolio_plan.json")
OUTPUT_PATH = Path("output/ibkr/contracts/opportunity-resolution-queue.json")
ASSET_CLASS_BY_TYPE = {
    "STOCK": "stock",
    "ETF": "etf",
    "BOND_ETF": "bond_etf",
    "COMMODITY_ETF": "commodity_etf",
    "COMMODITY_EQUITY_ETF": "commodity_etf",
    "COMMODITY_CLOSED_END_TRUST": "commodity_etf",
    "SUKUK_ETF": "etf",
}


def build_opportunity_contract_queue(project_root: Path) -> dict[str, Any]:
    funnel = _read_json(project_root / FUNNEL_PATH)
    screener = _read_json(project_root / SCREENER_PATH)
    active_plan = _read_json(project_root / ACTIVE_PLAN_PATH)
    blockers: list[str] = []
    if funnel is None:
        blockers.append("OPPORTUNITY_FUNNEL_REQUIRED")
    if screener is None:
        blockers.append("CURRENT_SCREENER_PREVIEW_REQUIRED")
    if blockers:
        return _publish(
            project_root,
            _report(
                status="SOURCE_ARTIFACT_BLOCKED",
                blockers=blockers,
                requests=[],
                skipped=[],
                source_hashes={},
            ),
        )

    assert funnel is not None
    assert screener is not None
    screener_by_symbol = {
        str(row.get("symbol", "")).upper(): row
        for row in screener.get("records", [])
        if isinstance(row, dict) and row.get("symbol")
    }
    metadata_by_symbol = broad_asset_metadata(project_root)
    eligible_rows = [
        row
        for row in funnel.get("watchlist_candidates", [])
        if isinstance(row, dict)
        and set(row.get("rejection_reasons", []))
        == {"CONTRACT_IDENTITY_REQUIRED"}
    ]
    active_opportunities = (
        active_plan.get("opportunities", {}).get("opportunities", [])
        if active_plan
        else []
    )
    for index, row in enumerate(active_opportunities, start=1):
        if not isinstance(row, dict):
            continue
        execution_blockers = set(row.get("execution_blockers", []))
        if execution_blockers != {"CONTRACT_IDENTITY_REQUIRED"}:
            continue
        symbol = str(row.get("ticker") or row.get("symbol") or "").upper()
        if not symbol or any(
            str(candidate.get("symbol", "")).upper() == symbol
            for candidate in eligible_rows
        ):
            continue
        eligible_rows.append(
            {
                "symbol": symbol,
                "rank": index,
                "asset_type": row.get("asset_type"),
                "rejection_reasons": ["CONTRACT_IDENTITY_REQUIRED"],
                "candidate_source": "CURRENT_ACTIVE_PORTFOLIO",
            }
        )
    eligible_rows.sort(
        key=lambda row: (
            int(row.get("rank", 10**9)),
            str(row.get("symbol", "")),
        )
    )
    requests: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate in eligible_rows[:50]:
        portfolio_symbol = str(candidate.get("symbol", "")).upper()
        screener_row = screener_by_symbol.get(portfolio_symbol, {})
        metadata = metadata_by_symbol.get(portfolio_symbol, {})
        broker_symbol = str(
            metadata.get("broker_symbol") or portfolio_symbol
        ).upper()
        asset_type = str(
            candidate.get("asset_type") or metadata.get("asset_type") or ""
        ).upper()
        asset_class = ASSET_CLASS_BY_TYPE.get(asset_type)
        currency = str(
            metadata.get("currency") or screener_row.get("currency") or ""
        ).upper()
        primary_exchange = str(
            metadata.get("primary_exchange")
            or screener_row.get("exchange")
            or ""
        ).upper()
        reasons: list[str] = []
        if asset_class is None:
            reasons.append("UNSUPPORTED_STK_ASSET_TYPE")
        if not currency:
            reasons.append("CURRENCY_REQUIRED")
        if not primary_exchange or primary_exchange == "SMART":
            reasons.append("PRIMARY_EXCHANGE_REQUIRED")
        if reasons:
            skipped.append(
                {
                    "symbol": portfolio_symbol,
                    "rank": candidate.get("rank"),
                    "reasons": reasons,
                }
            )
            continue
        request = {
                "symbol": broker_symbol,
                "asset_class": asset_class,
                "currency": currency,
                "exchange": "SMART",
                "primary_exchange": primary_exchange,
                "portfolio_rank": candidate.get("rank"),
                "queue_reason": "SOLE_BLOCKER_CONTRACT_IDENTITY_REQUIRED",
            }
        candidate_source = str(
            candidate.get("candidate_source")
            or "CURRENT_PORTFOLIO_WATCHLIST"
        )
        if candidate_source != "CURRENT_PORTFOLIO_WATCHLIST":
            request["candidate_source"] = candidate_source
        if broker_symbol != portfolio_symbol:
            request["portfolio_symbol"] = portfolio_symbol
        requests.append(request)

    status = "GO" if requests else "GO_EMPTY"
    return _publish(
        project_root,
        _report(
            status=status,
            blockers=[],
            requests=requests,
            skipped=skipped,
            source_hashes={
                "opportunity_funnel": funnel.get("content_hash")
                or stable_hash(funnel),
                "screener_preview": screener.get("content_hash")
                or stable_hash(screener),
                "active_portfolio_plan": (
                    active_plan.get("content_hash")
                    or stable_hash(active_plan)
                    if active_plan
                    else None
                ),
            },
        ),
    )


def _report(
    *,
    status: str,
    blockers: list[str],
    requests: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    source_hashes: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "ibkr_new_stock_contract_requests_v1",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes,
        "selection_contract": {
            "candidate_source": (
                "CURRENT_PORTFOLIO_WATCHLIST_OR_ACTIVE_PORTFOLIO"
            ),
            "required_only_blocker": "CONTRACT_IDENTITY_REQUIRED",
            "maximum_requests": 50,
            "request_exchange": "SMART",
            "primary_exchange_source": "CURRENT_SCREENER_EXCHANGE",
        },
        "blockers": sorted(set(blockers)),
        "requests": requests,
        "request_count": len(requests),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _publish(project_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    public = dict(payload)
    public["content_hash"] = stable_hash(public)
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(public, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return public


__all__ = ["build_opportunity_contract_queue"]
