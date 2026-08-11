from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from dotenv import dotenv_values

from stocks.domain.assets import AssetClass, IbkrSecurityType
from stocks.ibkr.connection import ReadOnlyIbkrConnectionService
from stocks.ibkr.contract_cache import (
    ContractCacheLayout,
    find_fresh_contract_cache_hit,
    read_contract_cache_rows,
    validate_contract_cache,
)
from stocks.ibkr.contract_resolver import LiveContractResolver
from stocks.ibkr.contracts import ContractResolutionRequest
from stocks.research.autopilot.contracts import stable_hash


LIVE_PORTS = {7496, 4001}
OUTPUT_PATH = Path("output/ibkr/contracts/live-read-only-refresh.json")
NEW_RESOLUTION_OUTPUT_PATH = Path(
    "output/ibkr/contracts/new-live-read-only-resolution.json"
)
NEW_REQUEST_MANIFEST_SCHEMA = "ibkr_new_stock_contract_requests_v1"
SUPPORTED_STK_ASSET_CLASSES = {
    AssetClass.STOCK,
    AssetClass.ETF,
    AssetClass.BOND_ETF,
    AssetClass.COMMODITY_ETF,
}


@dataclass(frozen=True)
class LiveReadOnlyContractSettings:
    env_file: Path
    host: str
    port: int
    client_id: int
    output_dir: Path
    connect_timeout_seconds: float = 12.0
    request_timeout_seconds: float = 15.0
    heartbeat_interval_seconds: float = 25.0
    stale_after_seconds: float = 45.0
    max_reconnect_attempts: int = 1
    reconnect_delays_seconds: tuple[float, ...] = (2.0,)

    def safe_dict(self) -> dict[str, Any]:
        return {
            "env_file": self.env_file.name,
            "host_local_only": self.host in {"127.0.0.1", "localhost"},
            "port_class": "LIVE" if self.port in LIVE_PORTS else "INVALID",
            "client_id": self.client_id,
            "read_only": True,
            "order_authority": "NONE",
            "allow_order_transmission": False,
            "live_trading_enabled": False,
        }


class ResolverProtocol(Protocol):
    def resolve(self, request: ContractResolutionRequest) -> Any: ...


ResolverFactory = Callable[
    [LiveReadOnlyContractSettings, ContractCacheLayout],
    ResolverProtocol,
]


def load_live_read_only_contract_settings(
    project_root: Path,
    env_file: str | Path = ".env.ibkr.live",
) -> tuple[LiveReadOnlyContractSettings | None, list[str]]:
    path = Path(env_file)
    if not path.is_absolute():
        path = project_root / path
    blockers: list[str] = []
    if path.name != ".env.ibkr.live" or not path.exists():
        return None, ["DEDICATED_LIVE_ENV_REQUIRED"]
    values = {
        key: str(value)
        for key, value in dotenv_values(path).items()
        if value is not None
    }
    host = values.get("IBKR_HOST", "")
    port = _integer(values.get("IBKR_PORT")) or -1
    if host not in {"127.0.0.1", "localhost"}:
        blockers.append("NON_LOCAL_IBKR_HOST_BLOCKED")
    if port not in LIVE_PORTS:
        blockers.append("LIVE_TWS_PORT_REQUIRED")
    if _boolean(values.get("IBKR_READ_ONLY")) is not True:
        blockers.append("READ_ONLY_MODE_REQUIRED")
    if values.get("IBKR_ORDER_AUTHORITY", "").upper() != "NONE":
        blockers.append("ORDER_AUTHORITY_NONE_REQUIRED")
    if _boolean(values.get("IBKR_ALLOW_ORDER_TRANSMISSION")) is not False:
        blockers.append("ORDER_TRANSMISSION_MUST_BE_DISABLED")
    if _boolean(values.get("IBKR_LIVE_TRADING_ENABLED")) is not False:
        blockers.append("LIVE_TRADING_MUST_BE_DISABLED_FOR_CONTRACT_REFRESH")

    configured_client_ids = {
        parsed
        for key, value in values.items()
        if key.endswith("CLIENT_ID")
        if (parsed := _integer(value)) is not None and parsed > 0
    }
    configured_refresh_id = values.get("IBKR_CONTRACT_REFRESH_CLIENT_ID")
    requested_client_id = (
        _integer(configured_refresh_id)
        if configured_refresh_id is not None
        else max(configured_client_ids, default=93) + 1
    )
    reserved_ids = {
        parsed
        for key, value in values.items()
        if key.endswith("CLIENT_ID") and key != "IBKR_CONTRACT_REFRESH_CLIENT_ID"
        if (parsed := _integer(value)) is not None and parsed > 0
    }
    if requested_client_id is None or requested_client_id <= 0:
        blockers.append("DEDICATED_CONTRACT_REFRESH_CLIENT_ID_REQUIRED")
    if requested_client_id is not None and requested_client_id in reserved_ids:
        blockers.append("CONTRACT_REFRESH_CLIENT_ID_COLLISION")
    if blockers:
        return None, sorted(set(blockers))
    assert requested_client_id is not None
    return (
        LiveReadOnlyContractSettings(
            env_file=path,
            host=host,
            port=port,
            client_id=requested_client_id,
            output_dir=project_root / "output" / "ibkr",
        ),
        [],
    )


def refresh_live_read_only_contracts(
    project_root: Path,
    *,
    symbols: list[str],
    env_file: str | Path = ".env.ibkr.live",
    resolver_factory: ResolverFactory | None = None,
) -> dict[str, Any]:
    requested_symbols = sorted(
        {symbol.strip().upper() for symbol in symbols if symbol.strip()}
    )
    settings, blockers = load_live_read_only_contract_settings(
        project_root,
        env_file,
    )
    if not requested_symbols:
        blockers.append("AT_LEAST_ONE_SYMBOL_REQUIRED")
    if len(requested_symbols) > 50:
        blockers.append("CONTRACT_REFRESH_SYMBOL_LIMIT_EXCEEDED")
    if settings is None or blockers:
        return _publish(
            project_root,
            _base_report(
                status="PREFLIGHT_BLOCKED",
                requested_symbols=requested_symbols,
                settings=settings,
                blockers=blockers,
            ),
        )

    layout = ContractCacheLayout.from_project_root(project_root)
    cache_validation = validate_contract_cache(layout)
    if cache_validation["status"] != "GO":
        return _publish(
            project_root,
            _base_report(
                status="CACHE_INVALID_BLOCKED",
                requested_symbols=requested_symbols,
                settings=settings,
                blockers=["PHASE2_CONTRACT_CACHE_INVALID"],
            ),
        )
    rows = read_contract_cache_rows(layout)
    stock_rows_by_symbol: dict[str, list[Any]] = {}
    for row in rows:
        if row.contract.security_type == IbkrSecurityType.STK:
            stock_rows_by_symbol.setdefault(row.contract.symbol.upper(), []).append(row)

    selection_blockers: list[str] = []
    selected_rows = []
    for symbol in requested_symbols:
        matches = stock_rows_by_symbol.get(symbol, [])
        if len(matches) != 1:
            selection_blockers.append(
                f"EXACT_EXISTING_STK_IDENTITY_REQUIRED:{symbol}"
            )
        else:
            selected_rows.append(matches[0])
    if selection_blockers:
        return _publish(
            project_root,
            _base_report(
                status="IDENTITY_SELECTION_BLOCKED",
                requested_symbols=requested_symbols,
                settings=settings,
                blockers=selection_blockers,
            ),
        )

    factory = resolver_factory or _default_resolver_factory
    resolver = factory(settings, layout)
    results: list[dict[str, Any]] = []
    read_only_calls = 0
    for row in selected_rows:
        contract = row.contract
        request = ContractResolutionRequest(
            symbol=contract.symbol,
            asset_class=AssetClass.STOCK,
            security_type=IbkrSecurityType.STK,
            currency=contract.currency,
            exchange=contract.exchange,
            primary_exchange=contract.primary_exchange,
        )
        resolved = resolver.resolve(request).as_dict()
        request_calls = int(
            resolved.get("read_only_ibkr_calls", {}).get(
                "req_contract_details",
                0,
            )
        )
        read_only_calls += request_calls
        identity = resolved.get("resolved_contract") or {}
        canonical_contract = identity.get("contract") or {}
        results.append(
            {
                "symbol": contract.symbol,
                "previous_con_id": contract.con_id,
                "status": resolved.get("status", "PROVIDER_ERROR"),
                "source": resolved.get("source", "UNAVAILABLE"),
                "cache_hit": bool(resolved.get("cache_hit")),
                "resolved_con_id": canonical_contract.get("conId"),
                "contract_hash": identity.get("contract_hash"),
                "resolved_at": identity.get("resolved_at"),
                "req_contract_details": request_calls,
                "financial_calls": resolved.get("financial_calls", {}),
            }
        )

    refreshed_rows = read_contract_cache_rows(layout)
    fresh_by_symbol = {
        row.contract.symbol.upper(): row
        for row in refreshed_rows
        if row.contract.security_type == IbkrSecurityType.STK
        and not row.is_stale()
    }
    freshness_failures = [
        symbol for symbol in requested_symbols if symbol not in fresh_by_symbol
    ]
    status = (
        "GO"
        if all(row["status"] == "RESOLVED" for row in results)
        and not freshness_failures
        else "NO_GO"
    )
    report = _base_report(
        status=status,
        requested_symbols=requested_symbols,
        settings=settings,
        blockers=[
            f"FRESH_CACHE_NOT_PROVEN:{symbol}"
            for symbol in freshness_failures
        ],
    )
    report.update(
        {
            "results": results,
            "refreshed_symbol_count": len(requested_symbols)
            - len(freshness_failures),
            "req_contract_details_calls": read_only_calls,
        }
    )
    return _publish(project_root, report)


def resolve_new_live_read_only_contracts(
    project_root: Path,
    *,
    manifest_file: str | Path,
    env_file: str | Path = ".env.ibkr.live",
    resolver_factory: ResolverFactory | None = None,
) -> dict[str, Any]:
    requests, manifest_metadata, manifest_blockers = _load_new_request_manifest(
        project_root,
        manifest_file,
    )
    settings, settings_blockers = load_live_read_only_contract_settings(
        project_root,
        env_file,
    )
    blockers = [*manifest_blockers, *settings_blockers]
    if blockers:
        return _publish(
            project_root,
            _new_resolution_report(
                status="PREFLIGHT_BLOCKED",
                requests=requests,
                manifest_metadata=manifest_metadata,
                settings=settings,
                blockers=blockers,
            ),
            path=NEW_RESOLUTION_OUTPUT_PATH,
        )

    layout = ContractCacheLayout.from_project_root(project_root)
    cache_validation = validate_contract_cache(layout)
    if cache_validation["status"] != "GO":
        return _publish(
            project_root,
            _new_resolution_report(
                status="CACHE_INVALID_BLOCKED",
                requests=requests,
                manifest_metadata=manifest_metadata,
                settings=settings,
                blockers=["PHASE2_CONTRACT_CACHE_INVALID"],
            ),
            path=NEW_RESOLUTION_OUTPUT_PATH,
        )

    assert settings is not None
    resolver = (resolver_factory or _default_resolver_factory)(settings, layout)
    results: list[dict[str, Any]] = []
    read_only_calls = 0
    write_counter_total = 0
    for request in requests:
        resolved = resolver.resolve(request).as_dict()
        request_calls = int(
            resolved.get("read_only_ibkr_calls", {}).get(
                "req_contract_details",
                0,
            )
        )
        financial_calls = {
            key: int(value)
            for key, value in resolved.get("financial_calls", {}).items()
        }
        request_write_calls = sum(financial_calls.values())
        read_only_calls += request_calls
        write_counter_total += request_write_calls
        identity = resolved.get("resolved_contract") or {}
        canonical_contract = identity.get("contract") or {}
        results.append(
            {
                "request": request.as_dict(),
                "status": resolved.get("status", "PROVIDER_ERROR"),
                "reason": resolved.get("reason"),
                "source": resolved.get("source", "UNAVAILABLE"),
                "cache_hit": bool(resolved.get("cache_hit")),
                "returned_match_count": int(
                    resolved.get("returned_match_count", 0)
                ),
                "resolved_con_id": canonical_contract.get("conId"),
                "resolved_primary_exchange": canonical_contract.get(
                    "primaryExchange"
                ),
                "contract_hash": identity.get("contract_hash"),
                "resolved_at": identity.get("resolved_at"),
                "req_contract_details": request_calls,
                "broker_write_calls": request_write_calls,
            }
        )
        if (
            resolved.get("status") == "PROVIDER_ERROR"
            and str(resolved.get("reason", "")).startswith(
                "IBKR connection is not healthy:"
            )
        ):
            remaining = requests[len(results) :]
            results.extend(
                {
                    "request": pending.as_dict(),
                    "status": "PROVIDER_ERROR",
                    "reason": "skipped after bounded connection failure",
                    "source": "ibkr_tws_live_read_only",
                    "cache_hit": False,
                    "returned_match_count": 0,
                    "resolved_con_id": None,
                    "resolved_primary_exchange": None,
                    "contract_hash": None,
                    "resolved_at": None,
                    "req_contract_details": 0,
                    "broker_write_calls": 0,
                }
                for pending in remaining
            )
            break

    cache_rows = read_contract_cache_rows(layout)
    fresh_exact_symbols: list[str] = []
    cache_proof_failures: list[str] = []
    for request in requests:
        try:
            fresh_hit = find_fresh_contract_cache_hit(cache_rows, request)
        except ValueError:
            fresh_hit = None
        if fresh_hit is None:
            cache_proof_failures.append(request.symbol)
        else:
            fresh_exact_symbols.append(request.symbol)

    result_failures = [
        row["request"]["symbol"]
        for row in results
        if row["status"] != "RESOLVED"
    ]
    output_blockers = [
        *(f"RESOLUTION_NOT_PROVEN:{symbol}" for symbol in result_failures),
        *(f"FRESH_EXACT_CACHE_NOT_PROVEN:{symbol}" for symbol in cache_proof_failures),
    ]
    if write_counter_total:
        output_blockers.append("BROKER_WRITE_COUNTER_NONZERO")
    status = "GO" if not output_blockers else "NO_GO"
    report = _new_resolution_report(
        status=status,
        requests=requests,
        manifest_metadata=manifest_metadata,
        settings=settings,
        blockers=output_blockers,
    )
    report.update(
        {
            "results": results,
            "resolved_symbol_count": len(fresh_exact_symbols),
            "fresh_exact_cache_symbols": sorted(fresh_exact_symbols),
            "req_contract_details_calls": read_only_calls,
            "broker_write_counter_total": write_counter_total,
        }
    )
    return _publish(
        project_root,
        report,
        path=NEW_RESOLUTION_OUTPUT_PATH,
    )


def _load_new_request_manifest(
    project_root: Path,
    manifest_file: str | Path,
) -> tuple[list[ContractResolutionRequest], dict[str, Any], list[str]]:
    path = Path(manifest_file)
    if not path.is_absolute():
        path = project_root / path
    resolved_root = project_root.resolve()
    resolved_path = path.resolve()
    metadata = {
        "manifest_file": path.name,
        "manifest_hash": None,
        "request_count": 0,
    }
    if not resolved_path.is_relative_to(resolved_root):
        return [], metadata, ["MANIFEST_OUTSIDE_PROJECT_ROOT_BLOCKED"]
    if not path.exists():
        return [], metadata, ["CONTRACT_REQUEST_MANIFEST_NOT_FOUND"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], metadata, ["CONTRACT_REQUEST_MANIFEST_INVALID_JSON"]
    metadata["manifest_hash"] = stable_hash(payload)
    if payload.get("schema") != NEW_REQUEST_MANIFEST_SCHEMA:
        return [], metadata, ["CONTRACT_REQUEST_MANIFEST_SCHEMA_INVALID"]
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        return [], metadata, ["AT_LEAST_ONE_EXPLICIT_REQUEST_REQUIRED"]
    if len(raw_requests) > 50:
        return [], metadata, ["CONTRACT_RESOLUTION_REQUEST_LIMIT_EXCEEDED"]

    requests: list[ContractResolutionRequest] = []
    blockers: list[str] = []
    seen_symbols: set[str] = set()
    for index, row in enumerate(raw_requests):
        prefix = f"REQUEST_{index}"
        if not isinstance(row, dict):
            blockers.append(f"{prefix}_OBJECT_REQUIRED")
            continue
        try:
            asset_class = AssetClass(str(row.get("asset_class", "stock")))
        except ValueError:
            blockers.append(f"{prefix}_ASSET_CLASS_INVALID")
            continue
        symbol = str(row.get("symbol", ""))
        primary_exchange = str(row.get("primary_exchange", ""))
        if asset_class not in SUPPORTED_STK_ASSET_CLASSES:
            blockers.append(f"{prefix}_STK_ASSET_CLASS_REQUIRED")
            continue
        if not primary_exchange:
            blockers.append(f"{prefix}_PRIMARY_EXCHANGE_REQUIRED")
            continue
        if symbol in seen_symbols:
            blockers.append(f"DUPLICATE_SYMBOL_BLOCKED:{symbol}")
            continue
        seen_symbols.add(symbol)
        request = ContractResolutionRequest(
            symbol=symbol,
            asset_class=asset_class,
            security_type=IbkrSecurityType.STK,
            currency=str(row.get("currency", "")),
            exchange=str(row.get("exchange", "")),
            primary_exchange=primary_exchange,
        )
        try:
            request.validate_basic()
        except ValueError:
            blockers.append(f"{prefix}_CONTRACT_REQUEST_INVALID")
            continue
        if request.exchange != "SMART":
            blockers.append(f"{prefix}_SMART_EXCHANGE_REQUIRED")
            continue
        requests.append(request)
    metadata["request_count"] = len(requests)
    return requests, metadata, sorted(set(blockers))


def _new_resolution_report(
    *,
    status: str,
    requests: list[ContractResolutionRequest],
    manifest_metadata: dict[str, Any],
    settings: LiveReadOnlyContractSettings | None,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "schema": "ibkr_new_live_read_only_contract_resolution_v1",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": manifest_metadata,
        "requests": [request.as_dict() for request in requests],
        "safe_connection": settings.safe_dict() if settings else None,
        "blockers": sorted(set(blockers)),
        "results": [],
        "resolved_symbol_count": 0,
        "fresh_exact_cache_symbols": [],
        "req_contract_details_calls": 0,
        "broker_write_counter_total": 0,
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
            "request_order_id": 0,
        },
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "execution_authority": "NONE",
        "live_trading_allowed": False,
        "credentials_logged": False,
        "account_identifiers_logged": False,
    }


def _default_resolver_factory(
    settings: LiveReadOnlyContractSettings,
    layout: ContractCacheLayout,
) -> LiveContractResolver:
    service = ReadOnlyIbkrConnectionService(settings)  # type: ignore[arg-type]
    return LiveContractResolver(
        service,
        layout,
        provider_source="ibkr_tws_live_read_only",
    )


def _base_report(
    *,
    status: str,
    requested_symbols: list[str],
    settings: LiveReadOnlyContractSettings | None,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "schema": "ibkr_live_read_only_contract_refresh_v1",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "requested_symbols": requested_symbols,
        "safe_connection": settings.safe_dict() if settings else None,
        "blockers": sorted(set(blockers)),
        "results": [],
        "refreshed_symbol_count": 0,
        "req_contract_details_calls": 0,
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
            "request_order_id": 0,
        },
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "execution_authority": "NONE",
        "live_trading_allowed": False,
        "credentials_logged": False,
        "account_identifiers_logged": False,
    }


def _publish(
    project_root: Path,
    payload: dict[str, Any],
    *,
    path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    public = dict(payload)
    public["content_hash"] = stable_hash(public)
    output_path = project_root / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(public, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return public


def _boolean(raw: str | None) -> bool | None:
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _integer(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


__all__ = [
    "LiveReadOnlyContractSettings",
    "load_live_read_only_contract_settings",
    "refresh_live_read_only_contracts",
    "resolve_new_live_read_only_contracts",
]
