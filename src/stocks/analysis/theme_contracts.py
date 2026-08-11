from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from stocks.domain.assets import AssetClass, IbkrSecurityType
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.connection import ReadOnlyIbkrConnectionService
from stocks.ibkr.contract_cache import (
    ContractCacheLayout,
    initialize_contract_cache,
    validate_contract_cache,
)
from stocks.ibkr.contract_resolver import LiveContractResolver
from stocks.ibkr.contracts import ContractResolutionRequest
from stocks.ibkr.live_contract_refresh import (
    LiveReadOnlyContractSettings,
    load_live_read_only_contract_settings,
)


CONFIG_PATH = Path("config/themes/frontier_technology_energy_v1.json")
OUTPUT_PATH = Path("output/analysis/themes/contract-coverage.json")
MAX_INSTRUMENTS = 30


class ResolverProtocol(Protocol):
    def resolve(self, request: ContractResolutionRequest) -> Any: ...


ResolverFactory = Callable[
    [LiveReadOnlyContractSettings, ContractCacheLayout],
    ResolverProtocol,
]


def collect_theme_contracts(
    project_root: Path,
    *,
    now: datetime | None = None,
    env_file: str | Path = ".env.ibkr.live",
    resolver_factory: ResolverFactory | None = None,
) -> dict[str, Any]:
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    config = _read_json(project_root / CONFIG_PATH)
    requests, config_blockers = _requests(config)
    settings, setting_blockers = load_live_read_only_contract_settings(
        project_root,
        env_file,
    )
    blockers = sorted(set(config_blockers + setting_blockers))
    if not requests:
        blockers.append("NO_THEME_CONTRACT_REQUESTS")
    if len(requests) > MAX_INSTRUMENTS:
        blockers.append("THEME_CONTRACT_REQUEST_LIMIT_EXCEEDED")
    if settings is None or blockers:
        return _publish(
            project_root,
            _base_report(
                observed_at,
                status="PREFLIGHT_BLOCKED",
                settings=settings,
                requested_count=len(requests),
                blockers=blockers,
            ),
        )

    layout = ContractCacheLayout.from_project_root(project_root)
    initialize_contract_cache(layout)
    cache_validation = validate_contract_cache(layout)
    if cache_validation.get("status") != "GO":
        return _publish(
            project_root,
            _base_report(
                observed_at,
                status="CACHE_INVALID_BLOCKED",
                settings=settings,
                requested_count=len(requests),
                blockers=["PHASE2_CONTRACT_CACHE_INVALID"],
            ),
        )

    factory = resolver_factory or _default_resolver_factory
    resolver = factory(settings, layout)
    results: list[dict[str, Any]] = []
    read_only_calls = 0
    for research_symbol, request in requests:
        resolved = resolver.resolve(request).as_dict()
        request_calls = int(
            (resolved.get("read_only_ibkr_calls") or {}).get(
                "req_contract_details",
                0,
            )
        )
        read_only_calls += request_calls
        identity = resolved.get("resolved_contract") or {}
        contract = identity.get("contract") or {}
        results.append(
            {
                "research_symbol": research_symbol,
                "ibkr_symbol": request.symbol,
                "status": resolved.get("status", "PROVIDER_ERROR"),
                "source": resolved.get("source", "UNAVAILABLE"),
                "cache_hit": bool(resolved.get("cache_hit")),
                "returned_match_count": int(
                    resolved.get("returned_match_count") or 0
                ),
                "reason": resolved.get("reason"),
                "contract_identity": (
                    {
                        "con_id": contract.get("conId"),
                        "symbol": contract.get("symbol"),
                        "local_symbol": contract.get("localSymbol"),
                        "security_type": contract.get("secType"),
                        "currency": contract.get("currency"),
                        "exchange": contract.get("exchange"),
                        "primary_exchange": contract.get("primaryExchange"),
                        "contract_hash": identity.get("contract_hash"),
                        "resolved_at": identity.get("resolved_at"),
                        "cache_ttl_seconds": identity.get(
                            "cache_ttl_seconds"
                        ),
                    }
                    if contract.get("conId")
                    else None
                ),
                "req_contract_details_calls": request_calls,
            }
        )

    resolved_count = sum(row["status"] == "RESOLVED" for row in results)
    status = (
        "GO"
        if resolved_count == len(results)
        else "PARTIAL"
        if resolved_count
        else "NO_GO"
    )
    report = _base_report(
        observed_at,
        status=status,
        settings=settings,
        requested_count=len(requests),
        blockers=[
            f"CONTRACT_NOT_RESOLVED:{row['research_symbol']}"
            for row in results
            if row["status"] != "RESOLVED"
        ],
    )
    report.update(
        {
            "resolved_count": resolved_count,
            "coverage_ratio": round(resolved_count / len(results), 6),
            "results": results,
            "provider_calls_read_only": read_only_calls,
            "req_contract_details_calls": read_only_calls,
        }
    )
    return _publish(project_root, report)


def _requests(
    config: dict[str, Any],
) -> tuple[list[tuple[str, ContractResolutionRequest]], list[str]]:
    output: list[tuple[str, ContractResolutionRequest]] = []
    blockers: list[str] = []
    seen: set[str] = set()
    for definition in (config.get("themes") or {}).values():
        for instrument in definition.get("instruments", []):
            research_symbol = str(instrument.get("symbol") or "").upper()
            spec = instrument.get("ibkr_contract") or {}
            ibkr_symbol = str(spec.get("symbol") or "").upper()
            if not research_symbol or not ibkr_symbol:
                blockers.append(
                    f"IBKR_CONTRACT_CONFIG_REQUIRED:{research_symbol or 'UNKNOWN'}"
                )
                continue
            if research_symbol in seen:
                blockers.append(f"DUPLICATE_THEME_SYMBOL:{research_symbol}")
                continue
            seen.add(research_symbol)
            try:
                asset_class = AssetClass(str(spec.get("asset_class")))
                request = ContractResolutionRequest(
                    symbol=ibkr_symbol,
                    asset_class=asset_class,
                    security_type=IbkrSecurityType.STK,
                    currency=str(spec.get("currency") or "").upper(),
                    exchange=str(spec.get("exchange") or "").upper(),
                    primary_exchange=(
                        str(spec["primary_exchange"]).upper()
                        if spec.get("primary_exchange")
                        else None
                    ),
                )
                request.validate_basic()
            except (TypeError, ValueError) as exc:
                blockers.append(
                    f"INVALID_THEME_CONTRACT_CONFIG:{research_symbol}:{exc}"
                )
                continue
            output.append((research_symbol, request))
    return output, blockers


def _default_resolver_factory(
    settings: LiveReadOnlyContractSettings,
    layout: ContractCacheLayout,
) -> LiveContractResolver:
    service = ReadOnlyIbkrConnectionService(settings)  # type: ignore[arg-type]
    return LiveContractResolver(
        service,
        layout,
        provider_source="ibkr_tws_live_read_only_theme_research",
    )


def _base_report(
    observed_at: datetime,
    *,
    status: str,
    settings: LiveReadOnlyContractSettings | None,
    requested_count: int,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "schema": "frontier_theme_contract_coverage_v1",
        "status": status,
        "generated_at": observed_at.isoformat(),
        "requested_count": requested_count,
        "resolved_count": 0,
        "coverage_ratio": 0.0,
        "safe_connection": settings.safe_dict() if settings else None,
        "blockers": sorted(set(blockers)),
        "results": [],
        "provider_calls_read_only": 0,
        "req_contract_details_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
            "request_order_id": 0,
        },
        "contract_identity_only": True,
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "live_trading_allowed": False,
        "credentials_logged": False,
        "account_identifiers_logged": False,
    }


def _publish(project_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    payload["content_hash"] = stable_hash(payload)
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


__all__ = ["collect_theme_contracts"]
