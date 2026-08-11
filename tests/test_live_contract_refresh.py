from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from stocks.domain.assets import IbkrSecurityType
from stocks.ibkr.contract_cache import (
    ContractCacheLayout,
    ContractCacheRow,
    contract_identity_document,
    read_contract_cache_rows,
    upsert_contract_cache_row,
    write_contract_cache_rows,
)
from stocks.ibkr.contract_resolver import LiveContractResolutionResult
from stocks.ibkr.contracts import (
    ContractResolutionStatus,
    ResolvedContract,
)
from stocks.ibkr.live_contract_refresh import (
    load_live_read_only_contract_settings,
    refresh_live_read_only_contracts,
    resolve_new_live_read_only_contracts,
)


def test_live_contract_refresh_settings_require_read_only_live_env(
    tmp_path: Path,
) -> None:
    env = _write_live_env(tmp_path)
    settings, blockers = load_live_read_only_contract_settings(tmp_path, env)

    assert blockers == []
    assert settings is not None
    assert settings.port == 7496
    assert settings.client_id == 94

    env.write_text(
        env.read_text(encoding="utf-8")
        .replace("IBKR_READ_ONLY=true", "IBKR_READ_ONLY=false")
        .replace("IBKR_ORDER_AUTHORITY=NONE", "IBKR_ORDER_AUTHORITY=CANARY")
        .replace(
            "IBKR_ALLOW_ORDER_TRANSMISSION=false",
            "IBKR_ALLOW_ORDER_TRANSMISSION=true",
        ),
        encoding="utf-8",
    )
    blocked, blockers = load_live_read_only_contract_settings(tmp_path, env)

    assert blocked is None
    assert "READ_ONLY_MODE_REQUIRED" in blockers
    assert "ORDER_AUTHORITY_NONE_REQUIRED" in blockers
    assert "ORDER_TRANSMISSION_MUST_BE_DISABLED" in blockers

    env.write_text(
        _live_env_text().replace(
            "IBKR_ALLOW_ORDER_TRANSMISSION=false",
            "IBKR_ALLOW_ORDER_TRANSMISSION=invalid",
        ),
        encoding="utf-8",
    )
    invalid, blockers = load_live_read_only_contract_settings(tmp_path, env)

    assert invalid is None
    assert "ORDER_TRANSMISSION_MUST_BE_DISABLED" in blockers


def test_refresh_updates_only_existing_exact_identity(tmp_path: Path) -> None:
    env = _write_live_env(tmp_path)
    layout = ContractCacheLayout.from_project_root(tmp_path)
    stale_row = ContractCacheRow(
        contract=_stock_contract(),
        resolved_at=datetime.now(UTC) - timedelta(days=8),
        server_version=224,
    )
    write_contract_cache_rows(layout, [stale_row])

    class FakeResolver:
        def resolve(self, request):
            fresh = ContractCacheRow(
                contract=stale_row.contract,
                resolved_at=datetime.now(UTC),
                server_version=225,
            )
            upsert_contract_cache_row(layout, fresh)
            return LiveContractResolutionResult(
                status=ContractResolutionStatus.RESOLVED,
                request=request,
                request_id=1,
                source="ibkr_tws_live_read_only",
                reason="unique exact match",
                returned_match_count=1,
                rejected_count=0,
                cache_hit=False,
                resolved_contract_identity=contract_identity_document(fresh),
                financial_calls={
                    "place_order": 0,
                    "cancel_order": 0,
                    "global_cancel": 0,
                },
                read_only_ibkr_calls={"req_contract_details": 1},
            )

    report = refresh_live_read_only_contracts(
        tmp_path,
        symbols=["AAPL"],
        env_file=env,
        resolver_factory=lambda _settings, _layout: FakeResolver(),
    )

    assert report["status"] == "GO"
    assert report["refreshed_symbol_count"] == 1
    assert report["req_contract_details_calls"] == 1
    assert report["financial_calls"]["place_order"] == 0
    assert report["execution_authority"] == "NONE"
    assert not read_contract_cache_rows(layout)[0].is_stale()


def test_refresh_blocks_symbol_not_already_in_contract_cache(
    tmp_path: Path,
) -> None:
    env = _write_live_env(tmp_path)
    layout = ContractCacheLayout.from_project_root(tmp_path)
    write_contract_cache_rows(
        layout,
        [
            ContractCacheRow(
                contract=_stock_contract(),
                resolved_at=datetime.now(UTC),
                server_version=225,
            )
        ],
    )

    report = refresh_live_read_only_contracts(
        tmp_path,
        symbols=["UNKNOWN"],
        env_file=env,
    )

    assert report["status"] == "IDENTITY_SELECTION_BLOCKED"
    assert report["req_contract_details_calls"] == 0
    assert report["financial_calls"]["place_order"] == 0


def test_new_resolution_persists_explicit_unique_stock_identities(
    tmp_path: Path,
) -> None:
    env = _write_live_env(tmp_path)
    manifest = _write_manifest(
        tmp_path,
        [
            _request("AMAT", "NASDAQ"),
            _request("ANET", "NYSE"),
        ],
    )
    layout = ContractCacheLayout.from_project_root(tmp_path)
    con_ids = {"AMAT": 101, "ANET": 102}

    class FakeResolver:
        def resolve(self, request):
            row = ContractCacheRow(
                contract=_stock_contract(
                    symbol=request.symbol,
                    con_id=con_ids[request.symbol],
                    primary_exchange=str(request.primary_exchange),
                ),
                resolved_at=datetime.now(UTC),
                server_version=225,
            )
            upsert_contract_cache_row(layout, row)
            return LiveContractResolutionResult(
                status=ContractResolutionStatus.RESOLVED,
                request=request,
                request_id=con_ids[request.symbol],
                source="ibkr_tws_live_read_only",
                reason="unique exact match",
                returned_match_count=1,
                rejected_count=0,
                cache_hit=False,
                resolved_contract_identity=contract_identity_document(row),
                financial_calls={
                    "place_order": 0,
                    "cancel_order": 0,
                    "global_cancel": 0,
                    "request_order_id": 0,
                },
                read_only_ibkr_calls={"req_contract_details": 1},
            )

    report = resolve_new_live_read_only_contracts(
        tmp_path,
        manifest_file=manifest,
        env_file=env,
        resolver_factory=lambda _settings, _layout: FakeResolver(),
    )

    assert report["status"] == "GO"
    assert report["resolved_symbol_count"] == 2
    assert report["fresh_exact_cache_symbols"] == ["AMAT", "ANET"]
    assert report["req_contract_details_calls"] == 2
    assert report["broker_write_counter_total"] == 0
    assert report["execution_authority"] == "NONE"
    assert {row.contract.symbol for row in read_contract_cache_rows(layout)} == {
        "AMAT",
        "ANET",
    }


def test_new_resolution_rejects_incomplete_manifest_before_resolver_creation(
    tmp_path: Path,
) -> None:
    env = _write_live_env(tmp_path)
    manifest = _write_manifest(
        tmp_path,
        [
            _request("AMAT", "NASDAQ"),
            _request("AMAT", "NASDAQ"),
            {
                "symbol": "ANET",
                "asset_class": "stock",
                "currency": "USD",
                "exchange": "SMART",
            },
        ],
    )

    def forbidden_factory(_settings, _layout):
        raise AssertionError("resolver must not be created for an invalid manifest")

    report = resolve_new_live_read_only_contracts(
        tmp_path,
        manifest_file=manifest,
        env_file=env,
        resolver_factory=forbidden_factory,
    )

    assert report["status"] == "PREFLIGHT_BLOCKED"
    assert "DUPLICATE_SYMBOL_BLOCKED:AMAT" in report["blockers"]
    assert "REQUEST_2_PRIMARY_EXCHANGE_REQUIRED" in report["blockers"]
    assert report["req_contract_details_calls"] == 0
    assert report["broker_write_counter_total"] == 0


def test_new_resolution_keeps_ambiguous_result_blocked_and_uncached(
    tmp_path: Path,
) -> None:
    env = _write_live_env(tmp_path)
    manifest = _write_manifest(tmp_path, [_request("ANET", "NYSE")])
    layout = ContractCacheLayout.from_project_root(tmp_path)

    class AmbiguousResolver:
        def resolve(self, request):
            return LiveContractResolutionResult(
                status=ContractResolutionStatus.AMBIGUOUS_BLOCKED,
                request=request,
                request_id=1,
                source="ibkr_tws_live_read_only",
                reason="more than one exact candidate",
                returned_match_count=2,
                rejected_count=0,
                cache_hit=False,
                financial_calls={
                    "place_order": 0,
                    "cancel_order": 0,
                    "global_cancel": 0,
                    "request_order_id": 0,
                },
                read_only_ibkr_calls={"req_contract_details": 1},
            )

    report = resolve_new_live_read_only_contracts(
        tmp_path,
        manifest_file=manifest,
        env_file=env,
        resolver_factory=lambda _settings, _layout: AmbiguousResolver(),
    )

    assert report["status"] == "NO_GO"
    assert report["results"][0]["status"] == "AMBIGUOUS_BLOCKED"
    assert report["resolved_symbol_count"] == 0
    assert report["broker_write_counter_total"] == 0
    assert read_contract_cache_rows(layout) == []


def _write_live_env(root: Path) -> Path:
    path = root / ".env.ibkr.live"
    path.write_text(_live_env_text(), encoding="utf-8")
    return path


def _write_manifest(root: Path, requests: list[dict[str, str]]) -> Path:
    path = root / "config" / "ibkr" / "new_stock_contract_requests_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "ibkr_new_stock_contract_requests_v1",
                "requests": requests,
            }
        ),
        encoding="utf-8",
    )
    return path


def _request(symbol: str, primary_exchange: str) -> dict[str, str]:
    return {
        "symbol": symbol,
        "asset_class": "stock",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": primary_exchange,
    }


def _live_env_text() -> str:
    return "\n".join(
        [
            "IBKR_HOST=127.0.0.1",
            "IBKR_PORT=7496",
            "IBKR_CLIENT_ID=91",
            "IBKR_RECON_CLIENT_ID=92",
            "IBKR_QUOTE_CLIENT_ID=93",
            "IBKR_READ_ONLY=true",
            "IBKR_ORDER_AUTHORITY=NONE",
            "IBKR_ALLOW_ORDER_TRANSMISSION=false",
            "IBKR_LIVE_TRADING_ENABLED=false",
        ]
    ) + "\n"


def _stock_contract(
    *,
    symbol: str = "AAPL",
    con_id: int = 265598,
    primary_exchange: str = "NASDAQ",
) -> ResolvedContract:
    return ResolvedContract(
        con_id=con_id,
        symbol=symbol,
        local_symbol=symbol,
        security_type=IbkrSecurityType.STK,
        exchange="SMART",
        currency="USD",
        trading_class="NMS",
        primary_exchange=primary_exchange,
        min_tick=Decimal("0.01"),
        valid_exchanges=f"SMART,{primary_exchange}",
        market_rule_ids="26,26",
        long_name=f"{symbol} INC",
        time_zone_id="US/Eastern",
        trading_hours="20260807:0400-20260807:2000",
        liquid_hours="20260807:0930-20260807:1600",
    )
