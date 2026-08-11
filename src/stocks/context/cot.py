from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from stocks.execution.idempotency import stable_hash


DISAGGREGATED_ENDPOINT = (
    "https://publicreporting.cftc.gov/resource/rxbv-e226.json"
)
TFF_ENDPOINT = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
REPORT_LAG = timedelta(days=3, hours=21)
MARKETS = {
    "GOLD": ("DISAGGREGATED", "088691"),
    "SILVER": ("DISAGGREGATED", "084691"),
    "COPPER": ("DISAGGREGATED", "085692"),
    "WTI_CRUDE": ("DISAGGREGATED", "067411"),
    "NATURAL_GAS": ("DISAGGREGATED", "023651"),
    "CORN": ("DISAGGREGATED", "002602"),
    "SOYBEANS": ("DISAGGREGATED", "005602"),
    "WHEAT": ("DISAGGREGATED", "001602"),
    "US_EQUITY_INDEX": ("TFF", "13874A"),
    "NASDAQ_100": ("TFF", "209742"),
    "USD_INDEX": ("TFF", "098662"),
    "UST_10Y": ("TFF", "043602"),
    "FED_FUNDS": ("TFF", "045601"),
}


def collect_cot_context(
    project_root: Path,
    *,
    start: str = "2018-01-01",
    observed_at: datetime | None = None,
    fetch: bool = True,
) -> dict[str, Any]:
    now = _utc(observed_at or datetime.now(UTC))
    output = project_root / "output" / "market_context"
    private = project_root / "data" / "market_context" / "private" / "cot"
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    snapshot_id = (
        f"COT-{now.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8].upper()}"
    )
    provider_errors: list[dict[str, str]] = []
    frames: list[pd.DataFrame] = []
    if fetch:
        for report_type, endpoint in (
            ("DISAGGREGATED", DISAGGREGATED_ENDPOINT),
            ("TFF", TFF_ENDPOINT),
        ):
            codes = [
                code
                for _, (kind, code) in MARKETS.items()
                if kind == report_type
            ]
            try:
                rows = _fetch_rows(
                    endpoint,
                    report_type=report_type,
                    codes=codes,
                    start=start,
                )
                frames.append(_normalize(rows, report_type=report_type))
            except Exception as exc:  # provider failure stays explicit
                provider_errors.append(
                    {
                        "report_type": report_type,
                        "error_class": type(exc).__name__,
                        "reason": str(exc)[:240],
                    }
                )
    else:
        frames.extend(_latest_private_frames(private))
    history = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    if not history.empty:
        history = (
            history.drop_duplicates(
                ["report_type", "cftc_contract_market_code", "report_date"],
                keep="last",
            )
            .sort_values(["market_id", "report_date"])
            .reset_index(drop=True)
        )
        snapshot_root = private / "snapshots" / f"snapshot_id={snapshot_id}"
        snapshot_root.mkdir(parents=True, exist_ok=False)
        history.to_parquet(snapshot_root / "history.parquet", index=False)
    contexts = _contexts(history, now=now)
    status = (
        "GO"
        if contexts and not provider_errors
        else "GO_DEGRADED"
        if contexts
        else "PROVIDER_UNAVAILABLE"
    )
    payload = {
        "schema": "cftc_cot_asset_context_v1",
        "status": status,
        "generated_at": now.isoformat(),
        "snapshot_id": snapshot_id if not history.empty else None,
        "source": "CFTC_PUBLIC_REPORTING_ENVIRONMENT",
        "source_urls": [DISAGGREGATED_ENDPOINT, TFF_ENDPOINT],
        "report_contract": (
            "Tuesday positions become available only after a conservative "
            "Friday publication lag"
        ),
        "report_lag_hours": REPORT_LAG.total_seconds() / 3600.0,
        "context_count": len(contexts),
        "contexts": contexts,
        "provider_errors": provider_errors,
        "standalone_entry_authority": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
    }
    payload["content_hash"] = stable_hash(payload)
    _write_json(output / "cot-context.json", payload)
    return payload


def cot_status(project_root: Path) -> dict[str, Any]:
    path = project_root / "output" / "market_context" / "cot-context.json"
    if not path.is_file():
        return {
            "schema": "cftc_cot_asset_context_v1",
            "status": "NOT_COLLECTED",
            "context_count": 0,
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "order_calls": 0,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategy_authority"] = "NONE"
    payload["execution_authority"] = "NONE"
    payload["broker_calls"] = 0
    payload["order_calls"] = 0
    return payload


def _fetch_rows(
    endpoint: str,
    *,
    report_type: str,
    codes: list[str],
    start: str,
) -> list[dict[str, Any]]:
    quoted_codes = ",".join(f"'{code}'" for code in codes)
    where = (
        f"report_date_as_yyyy_mm_dd >= '{start}T00:00:00.000' "
        f"AND cftc_contract_market_code IN({quoted_codes})"
    )
    if report_type == "DISAGGREGATED":
        where += " AND futonly_or_combined='FutOnly'"
    query = urllib.parse.urlencode(
        {
            "$limit": 50000,
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$where": where,
        }
    )
    request = urllib.request.Request(
        f"{endpoint}?{query}",
        headers={"User-Agent": "StocksResearch/1.0 read-only"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("CFTC_RESPONSE_NOT_A_LIST")
    return [row for row in payload if isinstance(row, dict)]


def _normalize(
    rows: list[dict[str, Any]], *, report_type: str
) -> pd.DataFrame:
    reverse = {
        (kind, code): market_id
        for market_id, (kind, code) in MARKETS.items()
    }
    records: list[dict[str, Any]] = []
    for row in rows:
        code = str(row.get("cftc_contract_market_code") or "").strip()
        market_id = reverse.get((report_type, code))
        if market_id is None:
            continue
        report_date = pd.to_datetime(
            row.get("report_date_as_yyyy_mm_dd"), utc=True, errors="coerce"
        )
        if pd.isna(report_date):
            continue
        open_interest = _number(row.get("open_interest_all"))
        if report_type == "DISAGGREGATED":
            primary_long = _number(row.get("m_money_positions_long_all"))
            primary_short = _number(row.get("m_money_positions_short_all"))
            secondary_long = _number(row.get("prod_merc_positions_long"))
            secondary_short = _number(row.get("prod_merc_positions_short"))
            primary_name = "MANAGED_MONEY"
            secondary_name = "PRODUCER_MERCHANT"
        else:
            primary_long = _number(row.get("asset_mgr_positions_long"))
            primary_short = _number(row.get("asset_mgr_positions_short"))
            secondary_long = _number(row.get("lev_money_positions_long"))
            secondary_short = _number(row.get("lev_money_positions_short"))
            primary_name = "ASSET_MANAGER"
            secondary_name = "LEVERAGED_MONEY"
        records.append(
            {
                "market_id": market_id,
                "report_type": report_type,
                "cftc_contract_market_code": code,
                "contract_market_name": row.get("contract_market_name"),
                "commodity_name": row.get("commodity_name"),
                "report_date": report_date,
                "available_at": report_date + REPORT_LAG,
                "open_interest": open_interest,
                "primary_category": primary_name,
                "primary_long": primary_long,
                "primary_short": primary_short,
                "primary_net": primary_long - primary_short,
                "secondary_category": secondary_name,
                "secondary_long": secondary_long,
                "secondary_short": secondary_short,
                "secondary_net": secondary_long - secondary_short,
            }
        )
    return pd.DataFrame(records)


def _contexts(history: pd.DataFrame, *, now: datetime) -> list[dict[str, Any]]:
    if history.empty:
        return []
    usable = history.loc[
        pd.to_datetime(history["available_at"], utc=True) <= pd.Timestamp(now)
    ].copy()
    contexts: list[dict[str, Any]] = []
    for market_id, group in usable.groupby("market_id"):
        group = group.sort_values("report_date").tail(156).copy()
        latest = group.iloc[-1]
        oi = float(latest["open_interest"])
        primary_norm = (
            float(latest["primary_net"]) / oi if oi > 0 else 0.0
        )
        secondary_norm = (
            float(latest["secondary_net"]) / oi if oi > 0 else 0.0
        )
        historical = group["primary_net"].div(
            group["open_interest"].replace(0, pd.NA)
        ).dropna()
        percentile = (
            float((historical <= primary_norm).mean())
            if len(historical) >= 26
            else math.nan
        )
        score = (
            max(-1.0, min(1.0, 2.0 * percentile - 1.0))
            if math.isfinite(percentile)
            else 0.0
        )
        available_at = pd.Timestamp(latest["available_at"])
        age_days = max(
            0.0,
            (pd.Timestamp(now) - available_at).total_seconds() / 86400.0,
        )
        stale = age_days > 14.0
        contexts.append(
            {
                "market_id": str(market_id),
                "report_type": latest["report_type"],
                "cftc_contract_market_code": latest[
                    "cftc_contract_market_code"
                ],
                "contract_market_name": latest["contract_market_name"],
                "report_date": pd.Timestamp(latest["report_date"]).isoformat(),
                "available_at": available_at.isoformat(),
                "age_days": round(age_days, 6),
                "status": "STALE_CONTEXT_BLOCKED" if stale else "CONTEXT_AVAILABLE",
                "open_interest": int(oi),
                "primary_category": latest["primary_category"],
                "primary_net": int(latest["primary_net"]),
                "primary_net_share_oi": round(primary_norm, 8),
                "secondary_category": latest["secondary_category"],
                "secondary_net": int(latest["secondary_net"]),
                "secondary_net_share_oi": round(secondary_norm, 8),
                "history_observation_count": len(group),
                "primary_position_percentile": (
                    round(percentile, 8) if math.isfinite(percentile) else None
                ),
                "positioning_score": 0.0 if stale else round(score, 8),
                "confidence": (
                    0.0
                    if stale
                    else round(min(0.85, 0.35 + len(group) / 312.0), 8)
                ),
                "standalone_entry_authority": False,
                "execution_authority": "NONE",
            }
        )
    return sorted(contexts, key=lambda row: row["market_id"])


def _latest_private_frames(private: Path) -> list[pd.DataFrame]:
    paths = sorted(
        private.glob("snapshots/snapshot_id=*/history.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [pd.read_parquet(paths[0])] if paths else []


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=UTC)
        if value.tzinfo is None
        else value.astimezone(UTC)
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = ["collect_cot_context", "cot_status"]
