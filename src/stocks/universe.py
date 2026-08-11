from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


BROAD_UNIVERSE_PATH = Path(
    "config/universes/broad_multi_asset_v1.json"
)
SECURITY_MASTER_PATH = Path(
    "data/research/phase11_4/private/security-master.parquet"
)
UNIVERSE_OUTPUT_ROOT = Path("output/universe")
UNIVERSE_TABLE_PATH = UNIVERSE_OUTPUT_ROOT / "instruments.parquet"
MEGA_CAP_SYMBOLS = frozenset(
    {
        "AAPL",
        "AMZN",
        "AVGO",
        "GOOG",
        "GOOGL",
        "META",
        "MSFT",
        "NVDA",
        "TSLA",
    }
)
COMMODITY_PRODUCER_METADATA: dict[str, tuple[str, str]] = {
    "AEM": ("GOLD", "PRECIOUS_METALS_PRODUCER"),
    "ALB": ("LITHIUM", "BATTERY_MATERIALS_PRODUCER"),
    "BHP": ("DIVERSIFIED_METALS", "DIVERSIFIED_MINER"),
    "CCJ": ("URANIUM", "URANIUM_PRODUCER"),
    "CF": ("FERTILIZER", "FERTILIZER_PRODUCER"),
    "CLF": ("STEEL", "STEEL_PRODUCER"),
    "FCX": ("COPPER", "COPPER_PRODUCER"),
    "GOLD": ("GOLD", "PRECIOUS_METALS_PRODUCER"),
    "HL": ("SILVER", "PRECIOUS_METALS_PRODUCER"),
    "LAC": ("LITHIUM", "BATTERY_MATERIALS_PRODUCER"),
    "LEU": ("URANIUM", "NUCLEAR_FUEL_CYCLE_COMPANY"),
    "MOS": ("FERTILIZER", "FERTILIZER_PRODUCER"),
    "MP": ("RARE_EARTHS", "CRITICAL_MATERIALS_PRODUCER"),
    "NEM": ("GOLD", "PRECIOUS_METALS_PRODUCER"),
    "NTR": ("FERTILIZER", "FERTILIZER_PRODUCER"),
    "NUE": ("STEEL", "STEEL_PRODUCER"),
    "PAAS": ("SILVER", "PRECIOUS_METALS_PRODUCER"),
    "RIO": ("DIVERSIFIED_METALS", "DIVERSIFIED_MINER"),
    "SCCO": ("COPPER", "COPPER_PRODUCER"),
    "UEC": ("URANIUM", "URANIUM_PRODUCER"),
}


def broad_universe(project_root: Path) -> list[dict[str, str]]:
    path = project_root / BROAD_UNIVERSE_PATH
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("broad universe groups must be a list")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("broad universe group must be an object")
        instruments = group.get("instruments", {})
        if not isinstance(instruments, dict):
            raise ValueError("broad universe instruments must be an object")
        for raw_symbol, classification in instruments.items():
            symbol = str(raw_symbol).strip().upper()
            if not symbol:
                raise ValueError("broad universe symbol is required")
            if symbol in seen:
                raise ValueError(
                    f"duplicate broad universe symbol: {symbol}"
                )
            seen.add(symbol)
            metadata: dict[str, Any] = {}
            if isinstance(classification, dict):
                metadata = classification
                sector = metadata.get("sector")
                region = metadata.get(
                    "region", group.get("region", "GLOBAL")
                )
                if not sector:
                    raise ValueError(
                        f"broad universe sector is required: {symbol}"
                    )
            elif isinstance(classification, list):
                sector, region = classification
            else:
                sector = classification
                region = group.get("region", "GLOBAL")
            row = {
                "symbol": symbol,
                "group": str(group.get("group", "UNKNOWN")),
                "asset_type": str(
                    metadata.get("asset_type", group["asset_type"])
                ),
                "sleeve": str(
                    metadata.get("sleeve", group["sleeve"])
                ),
                "sector": str(sector),
                "region": str(region),
            }
            defaults = {
                "product_structure": "UNCLASSIFIED_FUND",
                "commodity_exposure_type": "UNSPECIFIED",
                "underlying_commodity": (
                    str(sector)
                    if row["asset_type"].startswith("COMMODITY_")
                    else "NONE"
                ),
                "provider_symbol": symbol,
                "broker_symbol": symbol,
                "primary_exchange": "UNRESOLVED",
                "currency": "USD",
            }
            for key, default in defaults.items():
                row[key] = str(metadata.get(key, default))
            rows.append(row)
    return rows


def broad_universe_symbols(project_root: Path) -> set[str]:
    return {row["symbol"] for row in broad_universe(project_root)}


def broad_etf_symbols(project_root: Path) -> set[str]:
    return {
        row["symbol"]
        for row in broad_universe(project_root)
        if row["asset_type"].endswith("ETF")
    }


def broad_commodity_symbols(project_root: Path) -> set[str]:
    return {
        row["symbol"]
        for row in broad_universe(project_root)
        if row["asset_type"].startswith("COMMODITY_")
    }


def broad_asset_metadata(
    project_root: Path,
) -> dict[str, dict[str, str]]:
    return {
        row["symbol"]: {
            key: row[key]
            for key in (
                "asset_type",
                "sleeve",
                "sector",
                "region",
                "product_structure",
                "commodity_exposure_type",
                "underlying_commodity",
                "provider_symbol",
                "broker_symbol",
                "primary_exchange",
                "currency",
            )
        }
        for row in broad_universe(project_root)
    }


def commodity_producer_metadata(symbol: str) -> dict[str, str]:
    metadata = COMMODITY_PRODUCER_METADATA.get(symbol.strip().upper())
    if metadata is None:
        return {}
    underlying, exposure_type = metadata
    return {
        "product_structure": "OPERATING_COMPANY_EQUITY",
        "commodity_exposure_type": exposure_type,
        "underlying_commodity": underlying,
    }


def broad_universe_status(project_root: Path) -> dict[str, Any]:
    rows = broad_universe(project_root)
    return {
        "schema": "broad_multi_asset_universe_status_v1",
        "status": "GO" if rows else "NO_DATA",
        "instrument_count": len(rows),
        "group_count": len({row["group"] for row in rows}),
        "asset_type_count": len({row["asset_type"] for row in rows}),
        "sector_count": len({row["sector"] for row in rows}),
        "region_count": len({row["region"] for row in rows}),
        "commodity_product_structure_count": len(
            {
                row["product_structure"]
                for row in rows
                if row["asset_type"].startswith("COMMODITY_")
            }
        ),
        "commodity_underlying_count": len(
            {
                row["underlying_commodity"]
                for row in rows
                if row["asset_type"].startswith("COMMODITY_")
            }
        ),
        "automatic_execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def refresh_discovery_universe(project_root: Path) -> dict[str, Any]:
    security_path = project_root / SECURITY_MASTER_PATH
    if not security_path.is_file():
        return _blocked("SECURITY_MASTER_MISSING")
    stocks = pd.read_parquet(security_path)
    required = {
        "security_id",
        "ticker",
        "name",
        "exchange",
        "category",
        "is_delisted",
        "sector",
        "industry",
        "currency",
    }
    if missing := required - set(stocks.columns):
        return _blocked(
            "SECURITY_MASTER_SCHEMA_INVALID",
            missing_fields=sorted(missing),
        )
    signal_symbols = _current_signal_symbols(project_root)
    live_symbols = _live_allowlisted_symbols(project_root)
    opportunities = _opportunity_metadata(project_root)
    stocks = stocks.copy()
    stocks["instrument_id"] = stocks["security_id"].astype(str)
    stocks["symbol"] = stocks["ticker"].astype(str).str.upper()
    stocks["instrument_type"] = "STOCK"
    stocks["asset_type"] = "STOCK"
    stocks["sleeve"] = "stock"
    stocks["region"] = "UNITED_STATES_LISTED"
    stocks["country"] = "UNKNOWN_ISSUER_DOMICILE"
    stocks["exposure_type"] = "OPERATING_COMPANY_EQUITY"
    producer_metadata = stocks["symbol"].map(COMMODITY_PRODUCER_METADATA)
    producer_mask = producer_metadata.notna()
    stocks.loc[producer_mask, "exposure_type"] = (
        "COMMODITY_PRODUCER_EQUITY"
    )
    stocks["product_structure"] = "OPERATING_COMPANY_EQUITY"
    stocks["commodity_exposure_type"] = "NONE"
    stocks["underlying_commodity"] = "NONE"
    stocks.loc[producer_mask, "commodity_exposure_type"] = (
        producer_metadata.loc[producer_mask].map(lambda value: value[1])
    )
    stocks.loc[producer_mask, "underlying_commodity"] = (
        producer_metadata.loc[producer_mask].map(lambda value: value[0])
    )
    stocks["provider_symbol"] = stocks["symbol"]
    stocks["broker_symbol"] = stocks["symbol"]
    stocks["primary_exchange"] = stocks["exchange"]
    stocks["discovery_source"] = "PHASE11_4_PIT_SECURITY_MASTER"
    stocks["active_listing"] = ~stocks["is_delisted"].astype(bool)
    stocks["signal_eligible"] = (
        stocks["active_listing"]
        & stocks["symbol"].isin(signal_symbols)
    )
    stocks["live_executable"] = (
        stocks["active_listing"]
        & stocks["symbol"].isin(live_symbols)
    )
    stocks["mega_cap"] = stocks["symbol"].isin(MEGA_CAP_SYMBOLS)
    stocks["metadata_status"] = "PIT_SECURITY_MASTER"
    stocks["compliance_status"] = stocks["symbol"].map(
        lambda value: opportunities.get(value, {}).get(
            "shariah_status", "UNSCREENED"
        )
    )
    stocks.loc[
        ~stocks["active_listing"], "compliance_status"
    ] = "BLOCKED_DELISTED"
    stocks["eligibility_status"] = "RESEARCH_ONLY"
    stocks.loc[
        ~stocks["active_listing"], "eligibility_status"
    ] = "DELISTED_BLOCKED"
    stocks.loc[
        stocks["signal_eligible"], "eligibility_status"
    ] = "SIGNAL_ELIGIBLE"
    stocks.loc[
        stocks["live_executable"], "eligibility_status"
    ] = "LIVE_ALLOWLISTED"

    curated = pd.DataFrame(
        [_curated_record(row) for row in broad_universe(project_root)]
    )
    if not curated.empty:
        curated["signal_eligible"] = curated["symbol"].isin(
            signal_symbols
        )
        curated["live_executable"] = curated["symbol"].isin(live_symbols)
        curated["eligibility_status"] = "RESEARCH_ONLY"
        curated.loc[
            curated["signal_eligible"], "eligibility_status"
        ] = "SIGNAL_ELIGIBLE"
        curated.loc[
            curated["live_executable"], "eligibility_status"
        ] = "LIVE_ALLOWLISTED"
        curated["compliance_status"] = curated["symbol"].map(
            lambda value: opportunities.get(value, {}).get(
                "shariah_status", "UNSCREENED"
            )
        )

    columns = [
        "instrument_id",
        "security_id",
        "symbol",
        "name",
        "instrument_type",
        "asset_type",
        "category",
        "exposure_type",
        "product_structure",
        "commodity_exposure_type",
        "underlying_commodity",
        "provider_symbol",
        "broker_symbol",
        "primary_exchange",
        "exchange",
        "currency",
        "region",
        "country",
        "sector",
        "industry",
        "sleeve",
        "active_listing",
        "is_delisted",
        "signal_eligible",
        "live_executable",
        "mega_cap",
        "metadata_status",
        "compliance_status",
        "eligibility_status",
        "discovery_source",
    ]
    frame = pd.concat(
        [stocks.reindex(columns=columns), curated.reindex(columns=columns)],
        ignore_index=True,
    )
    frame = frame.sort_values(
        ["instrument_type", "symbol", "instrument_id"]
    ).reset_index(drop=True)
    output = project_root / UNIVERSE_TABLE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(output, frame)
    report = _universe_summary(frame)
    report["generated_at"] = datetime.now(UTC).isoformat()
    report["source_security_master"] = str(SECURITY_MASTER_PATH)
    report["universe_table"] = str(UNIVERSE_TABLE_PATH)
    report["execution_authority"] = "NONE"
    report["broker_calls"] = 0
    report["orders_generated"] = 0
    _atomic_json(
        project_root / UNIVERSE_OUTPUT_ROOT / "status.json", report
    )
    _publish_dimension_reports(project_root, frame)
    return report


def discovery_universe_command(
    project_root: Path,
    command: str,
) -> dict[str, Any]:
    if command == "refresh":
        return refresh_discovery_universe(project_root)
    frame = _load_discovery_universe(project_root)
    if frame is None:
        refreshed = refresh_discovery_universe(project_root)
        if refreshed.get("status") != "GO":
            return refreshed
        frame = _load_discovery_universe(project_root)
    assert frame is not None
    if command in {"status", "coverage"}:
        report = _universe_summary(frame)
        report["command"] = command
        report["execution_authority"] = "NONE"
        report["broker_calls"] = 0
        report["orders_generated"] = 0
        return report
    dimension_map = {
        "stocks": ("instrument_type", {"STOCK"}),
        "etfs": ("instrument_type", {"ETF"}),
        "commodities": (
            "instrument_type",
            {"COMMODITY_EXPOSURE"},
        ),
    }
    if command in dimension_map:
        column, allowed = dimension_map[command]
        selected = frame.loc[frame[column].isin(allowed)].copy()
        return _instrument_summary(command, selected)
    if command in {"sectors", "industries", "regions"}:
        dimensions = {
            "sectors": "sector",
            "industries": "industry",
            "regions": "region",
        }
        return rank_universe_dimension(
            project_root, dimensions[command]
        )
    return _blocked("UNKNOWN_UNIVERSE_COMMAND")


def rank_universe_dimension(
    project_root: Path,
    dimension: str,
) -> dict[str, Any]:
    if dimension not in {"sector", "industry", "region"}:
        return _blocked("UNKNOWN_RANKING_DIMENSION")
    frame = _load_discovery_universe(project_root)
    if frame is None:
        refreshed = refresh_discovery_universe(project_root)
        if refreshed.get("status") != "GO":
            return refreshed
        frame = _load_discovery_universe(project_root)
    assert frame is not None
    opportunities = _opportunity_metadata(project_root)
    opportunity_rows = []
    for symbol, row in opportunities.items():
        opportunity_rows.append(
            {
                "symbol": symbol,
                "opportunity_score": float(
                    row.get("opportunity_score", 0.0)
                ),
            }
        )
    scores = pd.DataFrame(opportunity_rows)
    work = frame.copy()
    if not scores.empty:
        work = work.merge(scores, on="symbol", how="left")
    else:
        work["opportunity_score"] = 0.0
    work["opportunity_score"] = work["opportunity_score"].fillna(0.0)
    work[dimension] = (
        work[dimension].fillna("UNKNOWN").astype(str)
    )
    groups: list[dict[str, Any]] = []
    for value, group in work.groupby(dimension, observed=True):
        scored = group.loc[group["opportunity_score"] > 0]
        groups.append(
            {
                dimension: value,
                "instrument_count": len(group),
                "active_listing_count": int(
                    group["active_listing"].sum()
                ),
                "signal_eligible_count": int(
                    group["signal_eligible"].sum()
                ),
                "live_executable_count": int(
                    group["live_executable"].sum()
                ),
                "signal_count": len(scored),
                "average_opportunity_score": round(
                    float(scored["opportunity_score"].mean())
                    if not scored.empty
                    else 0.0,
                    6,
                ),
                "maximum_opportunity_score": round(
                    float(scored["opportunity_score"].max())
                    if not scored.empty
                    else 0.0,
                    6,
                ),
            }
        )
    groups.sort(
        key=lambda row: (
            -float(row["maximum_opportunity_score"]),
            -int(row["signal_eligible_count"]),
            str(row[dimension]),
        )
    )
    return {
        "schema": f"{dimension}_ranking_v1",
        "status": "GO",
        "dimension": dimension,
        "count": len(groups),
        "rankings": [
            {"rank": index, **row}
            for index, row in enumerate(groups, start=1)
        ],
        "ranking_method": (
            "CURRENT_OPPORTUNITY_SCORE_THEN_SIGNAL_COVERAGE"
        ),
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _curated_record(row: dict[str, str]) -> dict[str, Any]:
    asset_type = row["asset_type"]
    if asset_type.startswith("COMMODITY_"):
        instrument_type = "COMMODITY_EXPOSURE"
        exposure_type = row["commodity_exposure_type"]
        if exposure_type == "UNSPECIFIED":
            exposure_type = _commodity_exposure(row["sector"])
    else:
        instrument_type = "ETF"
        exposure_type = _etf_exposure(asset_type)
    return {
        "instrument_id": f"CURATED:{row['symbol']}",
        "security_id": None,
        "symbol": row["symbol"],
        "name": row["symbol"],
        "instrument_type": instrument_type,
        "asset_type": asset_type,
        "category": row["group"],
        "exposure_type": exposure_type,
        "product_structure": row["product_structure"],
        "commodity_exposure_type": row["commodity_exposure_type"],
        "underlying_commodity": row["underlying_commodity"],
        "provider_symbol": row["provider_symbol"],
        "broker_symbol": row["broker_symbol"],
        "primary_exchange": row["primary_exchange"],
        "exchange": row["primary_exchange"],
        "currency": row["currency"],
        "region": row["region"],
        "country": _country_for_region(row["region"]),
        "sector": row["sector"],
        "industry": row["sector"],
        "sleeve": row["sleeve"],
        "active_listing": True,
        "is_delisted": False,
        "signal_eligible": False,
        "live_executable": False,
        "mega_cap": False,
        "metadata_status": "CURATED_RESEARCH_OVERLAY",
        "compliance_status": "UNSCREENED",
        "eligibility_status": "RESEARCH_ONLY",
        "discovery_source": "BROAD_MULTI_ASSET_V1",
    }


def _commodity_exposure(sector: str) -> str:
    mapping = {
        "GOLD": "GOLD_PRICE_EXPOSURE",
        "SILVER": "SILVER_PRICE_EXPOSURE",
        "OIL": "OIL_FUTURES_ROLL_EXPOSURE",
        "NATURAL_GAS": "NATURAL_GAS_FUTURES_ROLL_EXPOSURE",
        "COPPER": "COPPER_PRICE_EXPOSURE",
        "AGRICULTURE": "AGRICULTURE_FUTURES_BASKET_EXPOSURE",
        "INDUSTRIAL_METALS": "INDUSTRIAL_METALS_BASKET_EXPOSURE",
        "BROAD_COMMODITY": "BROAD_FUTURES_ROLL_EXPOSURE",
    }
    return mapping.get(sector, "COMMODITY_PRICE_EXPOSURE")


def _etf_exposure(asset_type: str) -> str:
    return {
        "BOND_ETF": "BOND_FUND_EXPOSURE",
        "REAL_ASSET_ETF": "REAL_ASSET_FUND_EXPOSURE",
        "CURRENCY_ETF": "CURRENCY_REFERENCE_EXPOSURE",
    }.get(asset_type, "EQUITY_FUND_EXPOSURE")


def _country_for_region(region: str) -> str:
    countries = {
        "AUSTRALIA",
        "CANADA",
        "CHINA",
        "FRANCE",
        "GERMANY",
        "INDIA",
        "JAPAN",
        "SOUTH_AFRICA",
        "SOUTH_KOREA",
        "TAIWAN",
        "UNITED_KINGDOM",
        "UNITED_STATES",
    }
    return region if region in countries else "MULTI_COUNTRY_OR_GLOBAL"


def _current_signal_symbols(project_root: Path) -> set[str]:
    payload = _read_json(
        project_root / "output/signals/latest_signals.json"
    )
    return {
        str(row.get("ticker", "")).upper()
        for row in payload.get("signals", [])
        if row.get("ticker")
    }


def _live_allowlisted_symbols(project_root: Path) -> set[str]:
    payload = _read_json(
        project_root / "output/ibkr/live/strategy-allowlist.json"
    )
    return {
        str(symbol).upper()
        for row in payload.get("strategies", [])
        if row.get("status") == "PIT_LIVE_ALLOWLISTED"
        for symbol in row.get("allowed_symbols", [])
    }


def _opportunity_metadata(
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    payload = _read_json(
        project_root / "output/portfolio/opportunity_ranking.json"
    )
    return {
        str(row.get("ticker", "")).upper(): row
        for row in payload.get("opportunities", [])
        if row.get("ticker")
    }


def _load_discovery_universe(
    project_root: Path,
) -> pd.DataFrame | None:
    path = project_root / UNIVERSE_TABLE_PATH
    return pd.read_parquet(path) if path.is_file() else None


def _universe_summary(frame: pd.DataFrame) -> dict[str, Any]:
    active_stocks = frame.loc[
        (frame["instrument_type"] == "STOCK")
        & frame["active_listing"].astype(bool)
    ]
    duplicate_symbols = Counter(frame["symbol"].astype(str))
    return {
        "schema": "broad_discovery_universe_status_v1",
        "status": "GO",
        "discovery_instrument_count": len(frame),
        "active_listing_count": int(frame["active_listing"].sum()),
        "delisted_count": int(frame["is_delisted"].sum()),
        "signal_eligible_instrument_count": int(
            frame["signal_eligible"].sum()
        ),
        "live_executable_instrument_count": int(
            frame["live_executable"].sum()
        ),
        "country_count": int(frame["country"].nunique()),
        "region_count": int(frame["region"].nunique()),
        "sector_count": int(frame["sector"].nunique()),
        "industry_count": int(frame["industry"].nunique()),
        "stock_count": int(
            (frame["instrument_type"] == "STOCK").sum()
        ),
        "etf_count": int((frame["instrument_type"] == "ETF").sum()),
        "commodity_exposure_count": int(
            (frame["instrument_type"] == "COMMODITY_EXPOSURE").sum()
        ),
        "commodity_linked_equity_count": int(
            (
                frame["exposure_type"]
                == "COMMODITY_PRODUCER_EQUITY"
            ).sum()
        ),
        "commodity_underlying_count": int(
            frame.loc[
                frame["underlying_commodity"] != "NONE",
                "underlying_commodity",
            ].nunique()
        ),
        "mega_cap_count": int(active_stocks["mega_cap"].sum()),
        "mega_cap_percentage_of_active_stocks": round(
            100
            * float(active_stocks["mega_cap"].mean())
            if not active_stocks.empty
            else 0.0,
            6,
        ),
        "duplicate_symbol_count": sum(
            count > 1 for count in duplicate_symbols.values()
        ),
        "point_in_time_security_master": True,
        "synthetic_instruments": 0,
        "automatic_execution_promotion": False,
    }


def _instrument_summary(
    command: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": f"universe_{command}_v1",
        "status": "GO",
        "instrument_type": command,
        "count": len(frame),
        "signal_eligible_count": int(frame["signal_eligible"].sum()),
        "live_executable_count": int(frame["live_executable"].sum()),
        "instruments": frame[
            [
                "symbol",
                "name",
                "asset_type",
                "exposure_type",
                "product_structure",
                "commodity_exposure_type",
                "underlying_commodity",
                "sector",
                "industry",
                "region",
                "country",
                "eligibility_status",
            ]
        ]
        .head(250)
        .to_dict(orient="records"),
        "result_truncated": len(frame) > 250,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _publish_dimension_reports(
    project_root: Path,
    frame: pd.DataFrame,
) -> None:
    del frame
    for dimension in ("sector", "industry", "region"):
        report = rank_universe_dimension(project_root, dimension)
        _atomic_json(
            project_root
            / UNIVERSE_OUTPUT_ROOT
            / f"{dimension}-ranking.json",
            report,
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": "broad_discovery_universe_status_v1",
        "status": "NO_GO",
        "blockers": [reason],
        **extra,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
