from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from stocks.domain.assets import IbkrSecurityType
from stocks.domain.currencies import validate_currency_code


_ID_PATTERN = re.compile(r"^[A-Z0-9_]+$")
_CODE_PATTERN = re.compile(r"^[A-Z0-9._-]+$")
_SLUG_PATTERN = re.compile(r"^[a-z0-9_]+$")

DEFAULT_RESEARCH_UNIVERSE: tuple[dict[str, Any], ...] = (
    {
        "instrument_id": "WORLD_EQUITY",
        "symbol": "ACWI",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "sleeve": "equity",
        "region": "global",
        "benchmark_role": True,
    },
    {
        "instrument_id": "US_EQUITY_CORE",
        "symbol": "SPY",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "equity",
        "region": "united_states",
        "benchmark_role": True,
    },
    {
        "instrument_id": "US_TECHNOLOGY",
        "symbol": "QQQ",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "sleeve": "equity",
        "region": "united_states",
        "benchmark_role": False,
    },
    {
        "instrument_id": "US_SMALL_CAP",
        "symbol": "IWM",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "equity",
        "region": "united_states",
        "benchmark_role": False,
    },
    {
        "instrument_id": "EUROPE_EQUITY",
        "symbol": "IEUR",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "equity",
        "region": "europe",
        "benchmark_role": False,
    },
    {
        "instrument_id": "JAPAN_EQUITY",
        "symbol": "EWJ",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "equity",
        "region": "japan",
        "benchmark_role": False,
    },
    {
        "instrument_id": "EMERGING_MARKETS",
        "symbol": "EEM",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "equity",
        "region": "emerging_markets",
        "benchmark_role": False,
    },
    {
        "instrument_id": "CHINA_EQUITY",
        "symbol": "MCHI",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "sleeve": "equity",
        "region": "china",
        "benchmark_role": False,
    },
    {
        "instrument_id": "INDIA_EQUITY",
        "symbol": "INDA",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "BATS",
        "sleeve": "equity",
        "region": "india",
        "benchmark_role": False,
    },
    {
        "instrument_id": "CASH_EQUIVALENT",
        "symbol": "BIL",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "cash",
        "region": "united_states",
        "benchmark_role": True,
    },
    {
        "instrument_id": "US_INTERMEDIATE_TREASURY",
        "symbol": "IEF",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "sleeve": "defensive",
        "region": "united_states",
        "benchmark_role": False,
    },
    {
        "instrument_id": "US_LONG_TREASURY",
        "symbol": "TLT",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "sleeve": "defensive",
        "region": "united_states",
        "benchmark_role": False,
    },
    {
        "instrument_id": "GOLD_DEFENSIVE",
        "symbol": "GLD",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "commodity",
        "region": "global",
        "benchmark_role": False,
    },
    {
        "instrument_id": "SILVER_COMMODITY",
        "symbol": "SLV",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "commodity",
        "region": "global",
        "benchmark_role": False,
    },
    {
        "instrument_id": "BROAD_COMMODITIES",
        "symbol": "DBC",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "commodity",
        "region": "global",
        "benchmark_role": False,
    },
    {
        "instrument_id": "INDUSTRIAL_METALS",
        "symbol": "DBB",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "commodity",
        "region": "global",
        "benchmark_role": False,
    },
    {
        "instrument_id": "AGRICULTURE_COMMODITIES",
        "symbol": "DBA",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "ARCA",
        "sleeve": "commodity",
        "region": "global",
        "benchmark_role": False,
    },
)


@dataclass(frozen=True)
class InstrumentManifestLayout:
    path: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> InstrumentManifestLayout:
        return cls(path=project_root / "data" / "instruments" / "research_universe.yaml")


def instrument_manifest_schema() -> dict[str, Any]:
    return {
        "schema": "instrument_manifest_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "required_fields": [
            "instrument_id",
            "symbol",
            "security_type",
            "currency",
            "exchange",
            "primary_exchange",
            "sleeve",
            "region",
            "benchmark_role",
        ],
        "supported_security_types": [item.value for item in IbkrSecurityType],
        "contract_validation_status": "UNVALIDATED_UNTIL_IBKR_CONTRACT_RESOLUTION",
        "financial_calls": _zero_financial_calls(),
    }


def default_instrument_manifest() -> dict[str, Any]:
    return {
        "schema": "instrument_manifest_v1",
        "contract_validation_status": "UNVALIDATED",
        "base_currency": "EUR",
        "notes": [
            "Symbols and exchanges are starting candidates only.",
            "Each row must be resolved by IBKR before it can seed bar collection.",
            "Futures are intentionally excluded until chain and roll rules are proven.",
        ],
        "instruments": [dict(item) for item in DEFAULT_RESEARCH_UNIVERSE],
        "financial_calls": _zero_financial_calls(),
    }


def initialize_instrument_manifest(layout: InstrumentManifestLayout) -> dict[str, Any]:
    manifest = default_instrument_manifest()
    validation = validate_instrument_manifest_payload(manifest)
    if validation["status"] != "GO":
        raise ValueError("default instrument manifest failed validation")
    layout.path.parent.mkdir(parents=True, exist_ok=True)
    layout.path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return {
        "schema": "instrument_manifest_init_command_v1",
        "status": "GO",
        "path": str(layout.path),
        "manifest": validation,
        "financial_calls": _zero_financial_calls(),
    }


def validate_instrument_manifest(layout: InstrumentManifestLayout) -> dict[str, Any]:
    if not layout.path.exists():
        return {
            "schema": "instrument_manifest_validation_v1",
            "status": "NO_MANIFEST",
            "path": str(layout.path),
            "instrument_count": 0,
            "errors": [f"{layout.path}: missing instrument manifest"],
            "financial_calls": _zero_financial_calls(),
        }
    try:
        payload = yaml.safe_load(layout.path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {
            "schema": "instrument_manifest_validation_v1",
            "status": "NO_GO",
            "path": str(layout.path),
            "instrument_count": 0,
            "errors": [f"{layout.path}: invalid YAML: {exc}"],
            "financial_calls": _zero_financial_calls(),
        }
    report = validate_instrument_manifest_payload(payload)
    report["path"] = str(layout.path)
    return report


def validate_instrument_manifest_payload(payload: object) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return _manifest_validation_report(["manifest payload must be an object"], [])
    if payload.get("schema") != "instrument_manifest_v1":
        errors.append("schema must be instrument_manifest_v1")
    if payload.get("contract_validation_status") != "UNVALIDATED":
        errors.append("contract_validation_status must remain UNVALIDATED until IBKR resolution")
    if payload.get("base_currency") != "EUR":
        errors.append("base_currency must be EUR")
    if payload.get("financial_calls") != _zero_financial_calls():
        errors.append("financial_calls must all be 0")
    instruments = payload.get("instruments")
    if not isinstance(instruments, list) or not instruments:
        errors.append("instruments must be a non-empty list")
        return _manifest_validation_report(errors, [])

    seen_ids: set[str] = set()
    seen_contract_requests: set[tuple[str, str, str, str, str]] = set()
    valid_instruments: list[dict[str, Any]] = []
    for index, item in enumerate(instruments):
        if not isinstance(item, dict):
            errors.append(f"instruments[{index}] must be an object")
            continue
        try:
            normalized = _validate_instrument_item(item, index)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if normalized["instrument_id"] in seen_ids:
            errors.append(f"instruments[{index}].instrument_id duplicates {normalized['instrument_id']}")
        seen_ids.add(normalized["instrument_id"])
        request_key = (
            normalized["symbol"],
            normalized["security_type"],
            normalized["currency"],
            normalized["exchange"],
            normalized["primary_exchange"],
        )
        if request_key in seen_contract_requests:
            errors.append(f"instruments[{index}] duplicates contract request {request_key}")
        seen_contract_requests.add(request_key)
        valid_instruments.append(normalized)
    return _manifest_validation_report(errors, valid_instruments)


def _validate_instrument_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    required = instrument_manifest_schema()["required_fields"]
    missing = [field for field in required if field not in item]
    if missing:
        raise ValueError(f"instruments[{index}] missing required fields: {', '.join(missing)}")
    instrument_id = _pattern_text(item["instrument_id"], _ID_PATTERN, f"instruments[{index}].instrument_id")
    symbol = _pattern_text(item["symbol"], _CODE_PATTERN, f"instruments[{index}].symbol")
    security_type = IbkrSecurityType(_pattern_text(item["security_type"], _CODE_PATTERN, f"instruments[{index}].security_type"))
    currency = _pattern_text(item["currency"], _CODE_PATTERN, f"instruments[{index}].currency")
    validate_currency_code(currency)
    exchange = _pattern_text(item["exchange"], _CODE_PATTERN, f"instruments[{index}].exchange")
    primary_exchange = _pattern_text(
        item["primary_exchange"],
        _CODE_PATTERN,
        f"instruments[{index}].primary_exchange",
    )
    sleeve = _pattern_text(item["sleeve"], _SLUG_PATTERN, f"instruments[{index}].sleeve")
    region = _pattern_text(item["region"], _SLUG_PATTERN, f"instruments[{index}].region")
    benchmark_role = item["benchmark_role"]
    if not isinstance(benchmark_role, bool):
        raise ValueError(f"instruments[{index}].benchmark_role must be boolean")
    return {
        "instrument_id": instrument_id,
        "symbol": symbol,
        "security_type": security_type.value,
        "currency": currency,
        "exchange": exchange,
        "primary_exchange": primary_exchange,
        "sleeve": sleeve,
        "region": region,
        "benchmark_role": benchmark_role,
    }


def _pattern_text(value: object, pattern: re.Pattern[str], field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    if not pattern.fullmatch(value):
        raise ValueError(f"{field_name} has invalid format")
    return value


def _manifest_validation_report(errors: list[str], instruments: list[dict[str, Any]]) -> dict[str, Any]:
    sleeve_counts: dict[str, int] = {}
    region_counts: dict[str, int] = {}
    benchmark_count = 0
    for item in instruments:
        sleeve_counts[item["sleeve"]] = sleeve_counts.get(item["sleeve"], 0) + 1
        region_counts[item["region"]] = region_counts.get(item["region"], 0) + 1
        if item["benchmark_role"]:
            benchmark_count += 1
    return {
        "schema": "instrument_manifest_validation_v1",
        "status": "GO" if not errors else "NO_GO",
        "contract_validation_status": "UNVALIDATED",
        "instrument_count": len(instruments),
        "benchmark_count": benchmark_count,
        "sleeve_counts": dict(sorted(sleeve_counts.items())),
        "region_counts": dict(sorted(region_counts.items())),
        "errors": errors,
        "financial_calls": _zero_financial_calls(),
    }


def _zero_financial_calls() -> dict[str, int]:
    return {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }
