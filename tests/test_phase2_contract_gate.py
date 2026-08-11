from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from ibapi.contract import Contract, ContractDetails

import main
from stocks.application.config import IbkrSettings
from stocks.application.phase_gates import phase1_freeze_status
from stocks.domain.assets import AssetClass, IbkrSecurityType
from stocks.ibkr.callbacks import CallbackState
from stocks.ibkr.connection import ReadOnlyIbkrConnectionService
from stocks.ibkr.contract_cache import (
    ContractCacheLayout,
    ContractCacheRow,
    FUTURE_CONTRACT_FIELDS,
    FUTURE_REFERENCE_METADATA_FIELDS,
    RESOLUTION_AUDIT_FIELDS,
    STOCK_CONTRACT_FIELDS,
    append_contract_error_audit,
    append_contract_request_audit,
    build_contract_resolution_artifacts,
    build_resolution_audit_record,
    canonical_contract_payload,
    contract_identity_document,
    contract_schema_manifest,
    contract_cache_ttl,
    export_contract_identity,
    contract_hash,
    empty_contract_manifest,
    find_fresh_contract_cache_hit,
    initialize_contract_cache,
    persist_contract_resolution_artifacts,
    read_contract_cache_records,
    read_contract_cache_rows,
    upsert_contract_cache_row,
    validate_unique_con_ids,
    validate_contract_cache,
    write_contract_cache_rows,
)
from stocks.ibkr.contract_resolver import LiveContractResolver
from stocks.ibkr.contracts import (
    ContractCandidateEvaluation,
    ContractRequestIdAllocator,
    ContractResolutionRequest,
    ContractResolutionStatus,
    FutureContractRequest,
    ResolvedContract,
    StockContractRequest,
    build_ibkr_contract_spec,
    build_native_ibapi_contract,
    classify_contract_match_count,
    contract_request_hash,
    evaluate_contract_candidates,
    evaluate_contract_details_payloads,
    gated_contract_resolution_report,
    resolved_contract_from_contract_details,
)
from stocks.ibkr.exchanges import parse_valid_exchanges
from stocks.ibkr.futures import validate_future_contract_month, validate_future_last_trade_time
from stocks.ibkr.market_rules import parse_market_rule_ids
from stocks.ibkr.trading_hours import parse_ibkr_hours
from stocks.ibkr.timezones import parse_ibkr_timezone_id


def test_phase1_gate_reports_missing_freeze_file(tmp_path) -> None:
    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False


def test_phase1_gate_rejects_marker_only_freeze_report(tmp_path) -> None:
    report_path = tmp_path / "PHASE1_FREEZE_REPORT.md"
    report_path.write_text(
        "IBKR_PHASE_1_READ_ONLY_CONNECTION_SERVICE_GO\nPHASE1_CONNECTION_SERVICE_FROZEN_GO\n",
        encoding="utf-8",
    )

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "verified artifact path is missing from PHASE1_FREEZE_REPORT.md"


def test_phase1_gate_accepts_freeze_report_with_verified_drill_artifact(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(tmp_path)
    _write_phase1_freeze_report(tmp_path, artifact_path)

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_FROZEN"
    assert status.frozen is True
    assert status.reason is None


def test_phase1_gate_rejects_freeze_report_with_artifact_hash_mismatch(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(tmp_path)
    _write_phase1_freeze_report(tmp_path, artifact_path, artifact_hash="A" * 64)

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "artifact SHA256 does not match PHASE1_FREEZE_REPORT.md"


def test_phase1_gate_rejects_freeze_report_with_wrong_artifact_name(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(tmp_path, file_name="disconnect_drill_go.json")
    _write_phase1_freeze_report(tmp_path, artifact_path)

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "verified artifact name must match phase1-disconnect-drill-YYYYMMDD-HHMMSS.json"


def test_phase1_gate_rejects_freeze_report_with_unparseable_artifact_timestamp(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(
        tmp_path,
        file_name="phase1-disconnect-drill-99999999-999999.json",
    )
    _write_phase1_freeze_report(tmp_path, artifact_path)

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "verified artifact name must match phase1-disconnect-drill-YYYYMMDD-HHMMSS.json"


def test_phase1_gate_rejects_freeze_report_with_reconnect_before_disconnect(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(
        tmp_path,
        observed_statuses=[
            {"phase": "start", "status": "HEALTHY"},
            {"phase": "reconnect", "status": "HEALTHY"},
            {"phase": "heartbeat", "status": "DISCONNECTED"},
        ],
    )
    _write_phase1_freeze_report(tmp_path, artifact_path)

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "verified artifact observed_statuses must include reconnect HEALTHY or DEGRADED after disconnect"


def test_phase1_gate_rejects_freeze_report_with_gateway_paper_port(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(tmp_path, port=4002)
    _write_phase1_freeze_report(tmp_path, artifact_path)

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "verified artifact port must be 7497 for the TWS paper Phase 1 freeze drill"


def test_phase1_gate_rejects_freeze_report_with_boolean_client_id(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(tmp_path, client_id=True)
    _write_phase1_freeze_report(tmp_path, artifact_path)

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "verified artifact client_id must be a positive integer"


def test_phase1_gate_rejects_freeze_report_with_boolean_financial_counter(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(
        tmp_path,
        financial_calls={
            "place_order": False,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    )
    _write_phase1_freeze_report(tmp_path, artifact_path)

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "verified artifact financial_calls.place_order must be 0"


def test_phase1_gate_rejects_freeze_report_after_phase1_file_changes(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(tmp_path)
    _write_phase1_freeze_report(tmp_path, artifact_path)
    (tmp_path / "src" / "stocks" / "ibkr" / "connection.py").write_text(
        "changed after freeze\n",
        encoding="utf-8",
    )

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "frozen hash mismatch for src/stocks/ibkr/connection.py"


def test_phase1_gate_rejects_freeze_report_after_immutable_config_changes(tmp_path) -> None:
    _write_required_phase1_files(tmp_path)
    artifact_path = _write_phase1_drill_artifact(tmp_path)
    _write_phase1_freeze_report(tmp_path, artifact_path)
    (tmp_path / "src" / "stocks" / "application" / "config.py").write_text(
        "changed immutable config after freeze\n",
        encoding="utf-8",
    )

    status = phase1_freeze_status(tmp_path)

    assert status.status == "PHASE1_NOT_FROZEN"
    assert status.frozen is False
    assert status.reason == "frozen hash mismatch for src/stocks/application/config.py"


def test_contract_resolution_is_blocked_before_phase1_freeze(tmp_path) -> None:
    phase1 = phase1_freeze_status(tmp_path)
    request = ContractResolutionRequest(
        symbol="SPY",
        asset_class=AssetClass.ETF,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="ARCA",
    )

    report = gated_contract_resolution_report(request, phase1).as_dict()

    assert report["status"] == "PHASE1_NOT_FROZEN"
    assert report["resolved_contract"] is None
    assert report["financial_calls"] == {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }


def test_contract_resolution_allows_future_chain_request_before_phase1_gate(tmp_path) -> None:
    phase1 = phase1_freeze_status(tmp_path)
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
    )

    report = gated_contract_resolution_report(request, phase1).as_dict()

    assert report["status"] == "PHASE1_NOT_FROZEN"
    assert report["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert report["reason"] == "Phase 2 contract resolution is blocked until Phase 1 freeze is proven."
    assert report["financial_calls"]["place_order"] == 0


def test_contract_cli_status_uses_phase_gate_without_tws(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "ibkr", "contract", "status"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "ibkr_contract_resolver_status_v1"
    assert payload["status"] == "PHASE1_NOT_FROZEN"
    assert payload["data_sources"]["external_providers"]["enabled"] is False
    assert payload["cache"]["schema"] == "ibkr_contract_cache_manifest_v1"
    assert payload["cache"]["financial_calls"] == {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }


def test_contract_cache_manifest_covers_required_phase2_fields(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    manifest = empty_contract_manifest(layout)

    assert manifest["files"]["stocks_parquet"].endswith("stocks.parquet")
    assert manifest["files"]["futures_parquet"].endswith("futures.parquet")
    assert "conId" in STOCK_CONTRACT_FIELDS
    assert "primaryExchange" in STOCK_CONTRACT_FIELDS
    assert "multiplier" in FUTURE_CONTRACT_FIELDS
    assert "liquidHours" in FUTURE_CONTRACT_FIELDS
    assert "contract_hash" in RESOLUTION_AUDIT_FIELDS


def test_contract_schema_manifest_contains_duckdb_contract_table() -> None:
    manifest = contract_schema_manifest()
    ddl = manifest["duckdb"]["contracts_ddl"]

    assert manifest["schema"] == "ibkr_contract_schema_v1"
    assert "CREATE TABLE IF NOT EXISTS ibkr_contracts" in ddl
    assert "con_id BIGINT PRIMARY KEY" in ddl
    assert "contract_hash VARCHAR NOT NULL" in ddl
    assert "delivery_type VARCHAR" in ddl
    assert "STK" in manifest["supported_security_types"]
    assert "FUT" in manifest["supported_security_types"]
    assert "deliveryType" in FUTURE_REFERENCE_METADATA_FIELDS
    assert manifest["optional_reference_fields"]["futures"] == list(FUTURE_REFERENCE_METADATA_FIELDS)
    assert manifest["financial_calls"] == {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }


def test_contract_cli_schema_prints_without_tws(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "ibkr", "contract", "schema"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "ibkr_contract_schema_v1"
    assert "ibkr_contracts" in payload["duckdb"]["contracts_ddl"]
    assert payload["financial_calls"]["place_order"] == 0


def test_contract_cache_ttl_policy_matches_phase2_gate() -> None:
    assert contract_cache_ttl(IbkrSecurityType.STK).days == 7
    assert contract_cache_ttl(IbkrSecurityType.FUT).total_seconds() == 24 * 60 * 60


def test_contract_hash_is_stable_for_canonical_contract() -> None:
    contract = ResolvedContract(
        con_id=756733,
        symbol="SPY",
        local_symbol="SPY",
        security_type=IbkrSecurityType.STK,
        exchange="SMART",
        primary_exchange="ARCA",
        currency="USD",
        trading_class="SPY",
        min_tick=Decimal("0.01"),
    )

    assert contract_hash(contract) == contract_hash(contract)
    assert len(contract_hash(contract)) == 64


def test_contract_match_count_classification_blocks_ambiguity() -> None:
    assert classify_contract_match_count(0) == ContractResolutionStatus.NOT_FOUND
    assert classify_contract_match_count(1) == ContractResolutionStatus.RESOLVED
    assert classify_contract_match_count(2) == ContractResolutionStatus.AMBIGUOUS_BLOCKED


def test_contract_cache_initialization_is_local_only(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    manifest = initialize_contract_cache(layout)

    assert manifest["initialized"] is True
    assert layout.cache_dir.exists()
    assert layout.requests_jsonl.exists()
    assert layout.errors_jsonl.exists()
    assert layout.manifest_json.exists()
    assert manifest["financial_calls"] == {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }


def test_contract_cli_init_cache_creates_manifest_without_tws(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "ibkr", "contract", "init-cache"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "ibkr_contract_cache_init_v1"
    assert payload["status"] == "GO"
    assert payload["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert (tmp_path / "output" / "ibkr" / "contracts" / "contract_manifest.json").exists()


def test_contract_cache_validation_accepts_initialized_manifest(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    initialize_contract_cache(layout)

    report = validate_contract_cache(layout)

    assert report["status"] == "GO"
    assert report["files"]["contract_manifest_json"]["exists"] is True
    assert report["files"]["contract_manifest_json"]["row_count"] == 1
    assert report["files"]["contract_manifest_json"]["error_count"] == 0


def test_contract_cache_validation_blocks_manifest_financial_calls(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    manifest = initialize_contract_cache(layout)
    manifest["financial_calls"]["place_order"] = 1
    layout.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert report["files"]["contract_manifest_json"]["error_count"] == 1
    assert any("financial_calls must all be 0" in error for error in report["errors"])


def test_contract_cache_validation_blocks_manifest_missing_optional_reference_fields(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    manifest = initialize_contract_cache(layout)
    del manifest["optional_reference_fields"]
    layout.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert report["files"]["contract_manifest_json"]["error_count"] == 1
    assert any("optional_reference_fields do not match Phase 2 schema" in error for error in report["errors"])


def test_contract_spec_for_us_stock_uses_ibkr_stk_fields() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )

    spec = build_ibkr_contract_spec(request).as_dict()

    assert spec["ibapi_fields"] == {
        "symbol": "AAPL",
        "secType": "STK",
        "exchange": "SMART",
        "currency": "USD",
        "primaryExchange": "NASDAQ",
    }
    assert spec["financial_calls"]["place_order"] == 0


def test_contract_spec_for_european_stock_keeps_primary_exchange() -> None:
    request = ContractResolutionRequest(
        symbol="ASML",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="EUR",
        exchange="SMART",
        primary_exchange="AEB",
    )

    spec = build_ibkr_contract_spec(request).as_ibapi_fields()

    assert spec["symbol"] == "ASML"
    assert spec["secType"] == "STK"
    assert spec["currency"] == "EUR"
    assert spec["primaryExchange"] == "AEB"


def test_contract_spec_for_etf_is_stk() -> None:
    request = ContractResolutionRequest(
        symbol="SPY",
        asset_class=AssetClass.ETF,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="ARCA",
    )

    spec = build_ibkr_contract_spec(request).as_ibapi_fields()

    assert spec["symbol"] == "SPY"
    assert spec["secType"] == "STK"
    assert spec["primaryExchange"] == "ARCA"


def test_contract_spec_for_future_requires_contract_month() -> None:
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
        expiry="202612",
    )

    spec = build_ibkr_contract_spec(request).as_ibapi_fields()

    assert spec == {
        "symbol": "GC",
        "secType": "FUT",
        "exchange": "COMEX",
        "currency": "USD",
        "lastTradeDateOrContractMonth": "202612",
    }


def test_contract_request_blocks_asset_class_security_type_mismatch() -> None:
    request = ContractResolutionRequest(
        symbol="SPY",
        asset_class=AssetClass.ETF,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="SMART",
        expiry="202612",
    )

    with pytest.raises(ValueError, match="requires secType STK"):
        request.validate_basic()


def test_contract_request_blocks_non_uppercase_currency_code() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="usd",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )

    with pytest.raises(ValueError, match="currency must be a 3-letter uppercase"):
        request.validate_basic()


def test_contract_request_blocks_currency_surrounding_whitespace() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency=" USD ",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )

    with pytest.raises(ValueError, match="currency must not contain leading or trailing whitespace"):
        request.validate_basic()


def test_contract_request_blocks_non_canonical_symbol_code() -> None:
    request = ContractResolutionRequest(
        symbol="aapl",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )

    with pytest.raises(ValueError, match="symbol must be an uppercase IBKR code"):
        request.validate_basic()


def test_contract_resolution_report_preserves_raw_invalid_request_symbol(tmp_path) -> None:
    phase1 = phase1_freeze_status(tmp_path)
    request = ContractResolutionRequest(
        symbol="aapl",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )

    report = gated_contract_resolution_report(request, phase1).as_dict()

    assert report["status"] == "VALIDATION_ERROR"
    assert report["request"]["symbol"] == "aapl"
    assert report["reason"] == "symbol must be an uppercase IBKR code"


def test_contract_request_blocks_exchange_surrounding_whitespace() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange=" SMART",
        primary_exchange="NASDAQ",
    )

    with pytest.raises(ValueError, match="exchange must not contain leading or trailing whitespace"):
        request.validate_basic()


def test_contract_request_blocks_non_canonical_primary_exchange() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="nasdaq",
    )

    with pytest.raises(ValueError, match="primary_exchange must be an uppercase IBKR code"):
        request.validate_basic()


def test_contract_spec_blocks_future_expiry_with_separator() -> None:
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
        expiry="2026-12",
    )

    with pytest.raises(ValueError, match="YYYYMM or YYYYMMDD"):
        build_ibkr_contract_spec(request)


def test_native_ibapi_contract_for_stock_sets_exact_fields() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )

    native_contract = build_native_ibapi_contract(build_ibkr_contract_spec(request))

    assert native_contract.symbol == "AAPL"
    assert native_contract.secType == "STK"
    assert native_contract.exchange == "SMART"
    assert native_contract.currency == "USD"
    assert native_contract.primaryExchange == "NASDAQ"
    assert native_contract.lastTradeDateOrContractMonth == ""


def test_native_ibapi_contract_for_future_sets_exact_contract_month() -> None:
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
        expiry="202612",
    )

    native_contract = build_native_ibapi_contract(build_ibkr_contract_spec(request))

    assert native_contract.symbol == "GC"
    assert native_contract.secType == "FUT"
    assert native_contract.exchange == "COMEX"
    assert native_contract.currency == "USD"
    assert native_contract.lastTradeDateOrContractMonth == "202612"
    assert native_contract.primaryExchange == ""


def test_resolved_contract_from_stock_contract_details_maps_required_fields() -> None:
    native_contract = Contract()
    native_contract.conId = 265598
    native_contract.symbol = "AAPL"
    native_contract.localSymbol = "AAPL"
    native_contract.secType = "STK"
    native_contract.exchange = "NASDAQ"
    native_contract.primaryExchange = "NASDAQ"
    native_contract.currency = "USD"
    native_contract.tradingClass = "NMS"
    details = ContractDetails()
    details.contract = native_contract
    details.minTick = 0.01
    details.validExchanges = "SMART,NASDAQ"
    details.marketRuleIds = "26"
    details.longName = "Apple Inc."
    details.industry = "Technology"
    details.category = "Computers"
    details.subcategory = "Computer Hardware"
    details.timeZoneId = "US/Eastern"
    details.tradingHours = "20260720:0930-1600"
    details.liquidHours = "20260720:0930-1600"

    resolved = resolved_contract_from_contract_details(details)

    assert resolved.con_id == 265598
    assert resolved.security_type == IbkrSecurityType.STK
    assert resolved.min_tick == Decimal("0.01")
    assert resolved.market_rule_ids == "26"
    resolved.validate_phase2_required_fields()


def test_resolved_contract_from_future_contract_details_maps_required_fields() -> None:
    native_contract = Contract()
    native_contract.conId = 999001
    native_contract.symbol = "GC"
    native_contract.localSymbol = "GCZ6"
    native_contract.secType = "FUT"
    native_contract.exchange = "COMEX"
    native_contract.currency = "USD"
    native_contract.tradingClass = "GC"
    native_contract.multiplier = "100"
    native_contract.lastTradeDateOrContractMonth = "202612"
    details = ContractDetails()
    details.contract = native_contract
    details.minTick = 0.1
    details.marketRuleIds = "332"
    details.realExpirationDate = "20261228"
    details.lastTradeTime = "12:30:00"
    details.underConId = 12345
    details.timeZoneId = "US/Eastern"
    details.tradingHours = "20260720:1800-1700"
    details.liquidHours = "20260720:1800-1700"

    resolved = resolved_contract_from_contract_details(details)

    assert resolved.con_id == 999001
    assert resolved.security_type == IbkrSecurityType.FUT
    assert resolved.expiry == "202612"
    assert resolved.multiplier == Decimal("1E+2")
    assert resolved.min_tick == Decimal("0.1")
    resolved.validate_phase2_required_fields()


def test_contract_details_payload_evaluation_resolves_exact_stock() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )

    evaluation = evaluate_contract_details_payloads(request, [_stock_contract_details("AAPL", 265598, "NASDAQ")])

    assert evaluation.status == ContractResolutionStatus.RESOLVED
    assert evaluation.matches[0].con_id == 265598
    assert evaluation.rejected_count == 0


def test_contract_details_payload_evaluation_blocks_ambiguous_symbol() -> None:
    request = ContractResolutionRequest(
        symbol="AIR",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="EUR",
        exchange="SMART",
    )

    evaluation = evaluate_contract_details_payloads(
        request,
        [
            _stock_contract_details("AIR", 111, "SBF", currency="EUR"),
            _stock_contract_details("AIR", 222, "AEB", currency="EUR"),
        ],
    )

    assert evaluation.status == ContractResolutionStatus.AMBIGUOUS_BLOCKED
    assert {match.con_id for match in evaluation.matches} == {111, 222}


def test_contract_details_payload_evaluation_reports_not_found_for_wrong_currency() -> None:
    request = ContractResolutionRequest(
        symbol="ASML",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="EUR",
        exchange="SMART",
        primary_exchange="AEB",
    )

    evaluation = evaluate_contract_details_payloads(
        request,
        [_stock_contract_details("ASML", 123456, "NASDAQ", currency="USD")],
    )

    assert evaluation.status == ContractResolutionStatus.NOT_FOUND
    assert evaluation.rejected_count == 1


def test_contract_details_payload_evaluation_counts_invalid_payload_rejection() -> None:
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
        expiry="202612",
    )

    evaluation = evaluate_contract_details_payloads(request, [object(), _future_contract_details("GC", 999001)])

    assert evaluation.status == ContractResolutionStatus.RESOLVED
    assert evaluation.matches[0].con_id == 999001
    assert evaluation.rejected_count == 1


def test_contract_spec_allows_future_chain_without_expiry() -> None:
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
    )

    spec = build_ibkr_contract_spec(request).as_ibapi_fields()

    assert spec == {
        "symbol": "GC",
        "secType": "FUT",
        "exchange": "COMEX",
        "currency": "USD",
    }


def test_contract_request_blocks_stock_expiry() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        expiry="202612",
    )

    with pytest.raises(ValueError, match="expiry is only valid for futures"):
        request.validate_basic()


def test_contract_cli_resolve_future_without_expiry_prepares_chain_request(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "ibkr",
            "contract",
            "resolve-future",
            "--symbol",
            "GC",
            "--exchange",
            "COMEX",
            "--currency",
            "USD",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "PHASE1_NOT_FROZEN"
    assert payload["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert payload["prepared_ibkr_contract_spec"]["ibapi_fields"] == {
        "symbol": "GC",
        "secType": "FUT",
        "exchange": "COMEX",
        "currency": "USD",
    }
    assert payload["financial_calls"]["place_order"] == 0


def test_contract_cli_resolve_stock_includes_prepared_contract_spec(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "ibkr",
            "contract",
            "resolve-stock",
            "--symbol",
            "AAPL",
            "--currency",
            "USD",
            "--exchange",
            "SMART",
            "--primary-exchange",
            "NASDAQ",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "PHASE1_NOT_FROZEN"
    assert payload["prepared_ibkr_contract_spec"]["ibapi_fields"]["secType"] == "STK"
    assert payload["prepared_ibkr_contract_spec"]["ibapi_fields"]["primaryExchange"] == "NASDAQ"
    assert payload["financial_calls"]["place_order"] == 0


def test_contract_cache_validation_accepts_empty_cache(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)

    report = validate_contract_cache(layout)

    assert report["schema"] == "ibkr_contract_cache_validation_v1"
    assert report["status"] == "GO"
    assert report["row_count"] == 0
    assert report["files"]["stocks_parquet"]["exists"] is False
    assert report["files"]["futures_parquet"]["exists"] is False
    assert report["financial_calls"]["place_order"] == 0


def test_contract_cli_validate_cache_runs_without_tws(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "ibkr", "contract", "validate-cache"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "ibkr_contract_cache_validation_command_v1"
    assert payload["status"] == "GO"
    assert payload["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert payload["cache_validation"]["row_count"] == 0
    assert payload["financial_calls"]["place_order"] == 0


def test_export_contract_identity_returns_not_found_for_empty_cache(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)

    export = export_contract_identity(layout, 265598)

    assert export["schema"] == "ibkr_contract_identity_export_v1"
    assert export["status"] == "NOT_FOUND"
    assert export["con_id"] == 265598
    assert export["resolved_contract_identity"] is None
    assert export["financial_calls"]["place_order"] == 0


def test_contract_cli_export_identity_reads_local_cache_without_tws(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    layout = ContractCacheLayout.from_project_root(tmp_path)
    contract = resolved_contract_from_contract_details(_stock_contract_details("AAPL", 265598, "NASDAQ"))
    write_contract_cache_rows(
        layout,
        [
            ContractCacheRow(
                contract=contract,
                resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
                server_version=225,
            )
        ],
    )

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "ibkr",
            "contract",
            "export-identity",
            "--con-id",
            "265598",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "ibkr_contract_identity_export_command_v1"
    assert payload["status"] == "GO"
    assert payload["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert payload["export"]["resolved_contract_identity"]["contract"]["conId"] == 265598
    assert payload["export"]["resolved_contract_identity"]["contract_hash"] == contract_hash(contract)
    assert payload["financial_calls"]["place_order"] == 0


def test_contract_cache_hit_returns_only_fresh_exact_match() -> None:
    request = ContractResolutionRequest(
        symbol="SPY",
        asset_class=AssetClass.ETF,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="ARCA",
    )
    contract = ResolvedContract(
        con_id=756733,
        symbol="SPY",
        local_symbol="SPY",
        security_type=IbkrSecurityType.STK,
        exchange="SMART",
        primary_exchange="ARCA",
        currency="USD",
        trading_class="SPY",
        min_tick=Decimal("0.01"),
    )
    now = datetime(2026, 7, 20, tzinfo=UTC)
    rows = [ContractCacheRow(contract=contract, resolved_at=now - timedelta(days=1), server_version=225)]

    hit = find_fresh_contract_cache_hit(rows, request, now=now)

    assert hit is rows[0]
    assert hit.as_resolution_audit(request, "CACHE_HIT")["selected_conId"] == 756733


def test_contract_cache_hit_blocks_invalid_request_before_matching() -> None:
    request = ContractResolutionRequest(
        symbol="spy",
        asset_class=AssetClass.ETF,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="ARCA",
    )
    contract = ResolvedContract(
        con_id=756733,
        symbol="SPY",
        local_symbol="SPY",
        security_type=IbkrSecurityType.STK,
        exchange="SMART",
        primary_exchange="ARCA",
        currency="USD",
        trading_class="SPY",
        min_tick=Decimal("0.01"),
    )
    now = datetime(2026, 7, 20, tzinfo=UTC)

    with pytest.raises(ValueError, match="symbol must be an uppercase IBKR code"):
        find_fresh_contract_cache_hit(
            [ContractCacheRow(contract=contract, resolved_at=now)],
            request,
            now=now,
        )


def test_contract_cache_stale_row_misses_for_refresh() -> None:
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
        expiry="202612",
    )
    contract = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
        min_tick=Decimal("0.1"),
        expiry="202612",
    )
    now = datetime(2026, 7, 20, tzinfo=UTC)
    rows = [ContractCacheRow(contract=contract, resolved_at=now - timedelta(hours=25))]

    assert find_fresh_contract_cache_hit(rows, request, now=now) is None


def test_contract_cache_duplicate_con_id_blocks() -> None:
    contract = ResolvedContract(
        con_id=123,
        symbol="AAA",
        local_symbol="AAA",
        security_type=IbkrSecurityType.STK,
        exchange="SMART",
        currency="USD",
        trading_class="AAA",
    )
    rows = [
        ContractCacheRow(contract=contract, resolved_at=datetime(2026, 7, 20, tzinfo=UTC)),
        ContractCacheRow(contract=contract, resolved_at=datetime(2026, 7, 20, tzinfo=UTC)),
    ]

    with pytest.raises(ValueError, match="duplicate conId"):
        validate_unique_con_ids(rows)


def test_contract_cache_ambiguous_request_match_blocks() -> None:
    request = ContractResolutionRequest(
        symbol="AIR",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="EUR",
        exchange="SMART",
    )
    now = datetime(2026, 7, 20, tzinfo=UTC)
    rows = [
        ContractCacheRow(
            contract=ResolvedContract(
                con_id=1,
                symbol="AIR",
                local_symbol="AIR",
                security_type=IbkrSecurityType.STK,
                exchange="SMART",
                currency="EUR",
                trading_class="AIR",
            ),
            resolved_at=now,
        ),
        ContractCacheRow(
            contract=ResolvedContract(
                con_id=2,
                symbol="AIR",
                local_symbol="AIR",
                security_type=IbkrSecurityType.STK,
                exchange="SMART",
                currency="EUR",
                trading_class="AIR",
            ),
            resolved_at=now,
        ),
    ]

    with pytest.raises(ValueError, match="ambiguous cache hit"):
        find_fresh_contract_cache_hit(rows, request, now=now)


def test_contract_candidate_evaluation_resolves_exact_stock_candidate() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )
    candidate = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
    )

    evaluation = evaluate_contract_candidates(request, [candidate])

    assert evaluation.status == ContractResolutionStatus.RESOLVED
    assert evaluation.as_dict()["selected_conId"] == 265598


def test_contract_candidate_evaluation_unknown_symbol_is_not_found() -> None:
    request = ContractResolutionRequest(
        symbol="UNKNOWN",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
    )

    evaluation = evaluate_contract_candidates(request, [])

    assert evaluation.status == ContractResolutionStatus.NOT_FOUND
    assert evaluation.as_dict()["returned_match_count"] == 0


def test_contract_candidate_evaluation_wrong_currency_blocks_match() -> None:
    request = ContractResolutionRequest(
        symbol="ASML",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="EUR",
        exchange="SMART",
        primary_exchange="AEB",
    )
    usd_candidate = ResolvedContract(
        con_id=123456,
        symbol="ASML",
        local_symbol="ASML",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="ASML",
    )

    evaluation = evaluate_contract_candidates(request, [usd_candidate])

    assert evaluation.status == ContractResolutionStatus.NOT_FOUND
    assert evaluation.rejected_count == 1


def test_contract_candidate_evaluation_wrong_exchange_blocks_match() -> None:
    request = ContractResolutionRequest(
        symbol="ASML",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="EUR",
        exchange="AEB",
        primary_exchange="AEB",
    )
    nasdaq_candidate = ResolvedContract(
        con_id=123456,
        symbol="ASML",
        local_symbol="ASML",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="EUR",
        trading_class="ASML",
    )

    evaluation = evaluate_contract_candidates(request, [nasdaq_candidate])

    assert evaluation.status == ContractResolutionStatus.NOT_FOUND
    assert evaluation.rejected_count == 1


def test_contract_candidate_evaluation_missing_future_multiplier_blocks_match() -> None:
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
        expiry="202612",
    )
    candidate = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        expiry="202612",
    )

    evaluation = evaluate_contract_candidates(request, [candidate])

    assert evaluation.status == ContractResolutionStatus.NOT_FOUND
    assert evaluation.rejected_count == 1


def test_contract_candidate_evaluation_missing_future_expiry_rejects_candidate_not_request() -> None:
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
    )
    candidate = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
    )

    evaluation = evaluate_contract_candidates(request, [candidate])

    assert evaluation.status == ContractResolutionStatus.NOT_FOUND
    assert evaluation.rejected_count == 1
    assert evaluation.rejected_candidates[0]["reasons"] == [
        "storage validation failed: resolved futures require expiry"
    ]


def test_contract_candidate_evaluation_ambiguous_candidates_block() -> None:
    request = ContractResolutionRequest(
        symbol="AIR",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="EUR",
        exchange="SMART",
    )
    candidates = [
        ResolvedContract(
            con_id=1,
            symbol="AIR",
            local_symbol="AIR",
            security_type=IbkrSecurityType.STK,
            exchange="SBF",
            currency="EUR",
            trading_class="AIR",
        ),
        ResolvedContract(
            con_id=2,
            symbol="AIR",
            local_symbol="AIR",
            security_type=IbkrSecurityType.STK,
            exchange="AEB",
            currency="EUR",
            trading_class="AIR",
        ),
    ]

    evaluation = evaluate_contract_candidates(request, candidates)

    assert evaluation.status == ContractResolutionStatus.AMBIGUOUS_BLOCKED
    assert len(evaluation.matches) == 2


def test_phase2_required_stock_fields_validate() -> None:
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
        valid_exchanges="SMART,NASDAQ",
        market_rule_ids="26",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        time_zone_id="US/Eastern",
        trading_hours="20260720:0930-1600",
        liquid_hours="20260720:0930-1600",
    )

    contract.validate_phase2_required_fields()


def test_phase2_required_fields_block_invalid_resolved_currency_code() -> None:
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="US1",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
        valid_exchanges="SMART,NASDAQ",
        market_rule_ids="26",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        time_zone_id="US/Eastern",
        trading_hours="20260720:0930-1600",
        liquid_hours="20260720:0930-1600",
    )

    with pytest.raises(ValueError, match="currency must be a 3-letter uppercase"):
        contract.validate_phase2_required_fields()


def test_ibkr_hours_parser_accepts_open_closed_and_prefixed_endpoints() -> None:
    windows = parse_ibkr_hours("20260720:0930-1600;20260721:CLOSED;20260722:0930-20260722:1600")

    assert len(windows) == 3
    assert windows[0].session_date.isoformat() == "2026-07-20"
    assert windows[0].start is not None
    assert windows[0].start.strftime("%H:%M") == "09:30"
    assert windows[0].closed is False
    assert windows[1].closed is True
    assert windows[2].end is not None
    assert windows[2].end.strftime("%H:%M") == "16:00"


def test_ibkr_timezone_parser_accepts_known_timezone_id() -> None:
    assert parse_ibkr_timezone_id("US/Eastern").key == "US/Eastern"


def test_phase2_required_fields_block_unknown_timezone_id() -> None:
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
        valid_exchanges="SMART,NASDAQ",
        market_rule_ids="26",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        time_zone_id="Not/AZone",
        trading_hours="20260720:0930-1600",
        liquid_hours="20260720:0930-1600",
    )

    with pytest.raises(ValueError, match="time_zone_id must be a known"):
        contract.validate_phase2_required_fields()


def test_phase2_required_fields_block_malformed_trading_hours() -> None:
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
        valid_exchanges="SMART,NASDAQ",
        market_rule_ids="26",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        time_zone_id="US/Eastern",
        trading_hours="20260720:BAD",
        liquid_hours="20260720:0930-1600",
    )

    with pytest.raises(ValueError, match="trading_hours must be parseable IBKR hours"):
        contract.validate_phase2_required_fields()


def test_phase2_required_stock_fields_block_missing_primary_exchange() -> None:
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
        valid_exchanges="SMART,NASDAQ",
        market_rule_ids="26",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        time_zone_id="US/Eastern",
        trading_hours="20260720:0930-1600",
        liquid_hours="20260720:0930-1600",
    )

    with pytest.raises(ValueError, match="primary_exchange"):
        contract.validate_phase2_required_fields()


def test_valid_exchanges_parser_accepts_comma_separated_exchange_codes() -> None:
    assert parse_valid_exchanges("SMART, NASDAQ, ARCA") == ("SMART", "NASDAQ", "ARCA")


def test_phase2_required_stock_fields_block_primary_exchange_absent_from_valid_exchanges() -> None:
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
        valid_exchanges="SMART,ARCA",
        market_rule_ids="26",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        time_zone_id="US/Eastern",
        trading_hours="20260720:0930-1600",
        liquid_hours="20260720:0930-1600",
    )

    with pytest.raises(ValueError, match="primary_exchange must be present in valid_exchanges"):
        contract.validate_phase2_required_fields()


def test_phase2_required_stock_fields_block_nonpositive_min_tick() -> None:
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0"),
        valid_exchanges="SMART,NASDAQ",
        market_rule_ids="26",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        time_zone_id="US/Eastern",
        trading_hours="20260720:0930-1600",
        liquid_hours="20260720:0930-1600",
    )

    with pytest.raises(ValueError, match="min_tick must be positive"):
        contract.validate_phase2_required_fields()


def test_phase2_required_future_fields_validate() -> None:
    contract = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
        min_tick=Decimal("0.1"),
        expiry="202612",
        market_rule_ids="332",
        last_trade_date_or_contract_month="202612",
        real_expiration_date="20261228",
        last_trade_time="12:30:00",
        under_con_id=12345,
        time_zone_id="US/Eastern",
        trading_hours="20260720:1800-1700",
        liquid_hours="20260720:1800-1700",
    )

    contract.validate_phase2_required_fields()


def test_future_reference_metadata_requires_complete_optional_set() -> None:
    contract = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
        min_tick=Decimal("0.1"),
        expiry="202612",
        market_rule_ids="332",
        last_trade_date_or_contract_month="202612",
        real_expiration_date="20261228",
        last_trade_time="12:30:00",
        under_con_id=12345,
        delivery_type="PHYSICAL",
        time_zone_id="US/Eastern",
        trading_hours="20260720:1800-1700",
        liquid_hours="20260720:1800-1700",
    )

    with pytest.raises(ValueError, match="incomplete FUT reference metadata"):
        contract.validate_phase2_required_fields()


def test_future_reference_metadata_is_not_allowed_on_stock() -> None:
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
        valid_exchanges="SMART,NASDAQ",
        market_rule_ids="26",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        delivery_type="PHYSICAL",
        settlement_type="DELIVERABLE",
        contract_size_unit="SHARES",
        roll_group="AAPL",
        time_zone_id="US/Eastern",
        trading_hours="20260720:0930-1600",
        liquid_hours="20260720:0930-1600",
    )

    with pytest.raises(ValueError, match="only valid for futures"):
        contract.validate_phase2_required_fields()


def test_future_lifecycle_validators_accept_contract_month_and_trade_time() -> None:
    validate_future_contract_month("202612", field_name="expiry")
    validate_future_contract_month("20261228", field_name="last_trade_date_or_contract_month")
    validate_future_last_trade_time("12:30")
    validate_future_last_trade_time("12:30:00")


def test_phase2_required_future_fields_block_invalid_real_expiration_date() -> None:
    contract = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
        min_tick=Decimal("0.1"),
        expiry="202612",
        market_rule_ids="332",
        last_trade_date_or_contract_month="202612",
        real_expiration_date="20261340",
        last_trade_time="12:30:00",
        under_con_id=12345,
        time_zone_id="US/Eastern",
        trading_hours="20260720:1800-1700",
        liquid_hours="20260720:1800-1700",
    )

    with pytest.raises(ValueError, match="real_expiration_date contains an invalid YYYYMMDD date"):
        contract.validate_phase2_required_fields()


def test_phase2_required_future_fields_block_nonpositive_under_con_id() -> None:
    contract = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
        min_tick=Decimal("0.1"),
        expiry="202612",
        market_rule_ids="332",
        last_trade_date_or_contract_month="202612",
        real_expiration_date="20261228",
        last_trade_time="12:30:00",
        under_con_id=0,
        time_zone_id="US/Eastern",
        trading_hours="20260720:1800-1700",
        liquid_hours="20260720:1800-1700",
    )

    with pytest.raises(ValueError, match="under_con_id must be positive"):
        contract.validate_phase2_required_fields()


def test_phase2_required_future_fields_block_missing_market_rule_ids() -> None:
    contract = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
        min_tick=Decimal("0.1"),
        expiry="202612",
        last_trade_date_or_contract_month="202612",
        real_expiration_date="20261228",
        last_trade_time="12:30:00",
        under_con_id=12345,
        time_zone_id="US/Eastern",
        trading_hours="20260720:1800-1700",
        liquid_hours="20260720:1800-1700",
    )

    with pytest.raises(ValueError, match="market_rule_ids"):
        contract.validate_phase2_required_fields()


def test_market_rule_ids_parser_accepts_comma_separated_positive_integers() -> None:
    assert parse_market_rule_ids("26, 332,1000") == (26, 332, 1000)


def test_phase2_required_fields_block_malformed_market_rule_ids() -> None:
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
        valid_exchanges="SMART,NASDAQ",
        market_rule_ids="26,BAD",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        time_zone_id="US/Eastern",
        trading_hours="20260720:0930-1600",
        liquid_hours="20260720:0930-1600",
    )

    with pytest.raises(ValueError, match="market_rule_ids must be comma-separated positive integers"):
        contract.validate_phase2_required_fields()


def test_resolution_audit_record_covers_required_fields_for_resolved_candidate(tmp_path) -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )
    candidate = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
    )
    evaluation = evaluate_contract_candidates(request, [candidate])
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)

    record = build_resolution_audit_record(
        request,
        evaluation,
        resolved_at=resolved_at,
        server_version=225,
    )

    assert set(RESOLUTION_AUDIT_FIELDS).issubset(record)
    assert record["selected_conId"] == 265598
    assert record["resolution_status"] == "RESOLVED"
    assert record["server_version"] == 225
    assert isinstance(record["contract_hash"], str)

    layout = ContractCacheLayout.from_project_root(tmp_path)
    append_contract_request_audit(layout, record)

    saved = json.loads(layout.requests_jsonl.read_text(encoding="utf-8"))
    assert saved["requested_symbol"] == "AAPL"
    assert saved["selected_conId"] == 265598


def test_resolution_audit_record_blocks_invalid_request_before_normalization() -> None:
    request = ContractResolutionRequest(
        symbol="aapl",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )
    evaluation = ContractCandidateEvaluation(
        status=ContractResolutionStatus.NOT_FOUND,
        matches=(),
        rejected_count=0,
        reason="no candidates",
    )

    with pytest.raises(ValueError, match="symbol must be an uppercase IBKR code"):
        build_resolution_audit_record(
            request,
            evaluation,
            resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
            server_version=225,
        )


def test_resolution_error_audit_record_writes_not_found_without_contract_hash(tmp_path) -> None:
    request = ContractResolutionRequest(
        symbol="MISSING",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
    )
    evaluation = evaluate_contract_candidates(request, [])
    layout = ContractCacheLayout.from_project_root(tmp_path)
    record = build_resolution_audit_record(
        request,
        evaluation,
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=225,
    )

    append_contract_error_audit(layout, record)

    saved = json.loads(layout.errors_jsonl.read_text(encoding="utf-8"))
    assert saved["resolution_status"] == "NOT_FOUND"
    assert saved["selected_conId"] is None
    assert saved["contract_hash"] is None


def test_contract_resolution_artifacts_build_audit_and_cache_row_from_payload_evaluation() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    evaluation = evaluate_contract_details_payloads(request, [_stock_contract_details("AAPL", 265598, "NASDAQ")])

    artifacts = build_contract_resolution_artifacts(
        request,
        evaluation,
        resolved_at=resolved_at,
        server_version=225,
    )

    assert artifacts.resolved is True
    assert artifacts.audit_record["resolution_status"] == "RESOLVED"
    assert artifacts.audit_record["selected_conId"] == 265598
    assert artifacts.cache_row is not None
    assert artifacts.cache_row.contract.con_id == 265598
    assert artifacts.cache_row.resolved_at == resolved_at
    assert artifacts.cache_row.server_version == 225
    assert artifacts.as_dict()["financial_calls"]["place_order"] == 0
    assert artifacts.as_dict()["resolved_contract_identity"]["contract"]["conId"] == 265598
    assert artifacts.as_dict()["resolved_contract_identity"]["contract_hash"] == artifacts.cache_row.contract_hash


def test_contract_identity_document_exports_stock_ibkr_fields() -> None:
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    contract = resolved_contract_from_contract_details(_stock_contract_details("AAPL", 265598, "NASDAQ"))
    row = ContractCacheRow(contract=contract, resolved_at=resolved_at, server_version=225)

    document = contract_identity_document(row)

    assert document["schema"] == "ibkr_contract_identity_v1"
    assert document["contract"]["conId"] == 265598
    assert document["contract"]["secType"] == "STK"
    assert document["contract"]["primaryExchange"] == "NASDAQ"
    assert document["contract"]["minTick"] == "0.01"
    assert document["resolved_at"] == "2026-07-20T00:00:00+00:00"
    assert document["contract_hash"] == contract_hash(contract)
    assert document["cache_ttl_seconds"] == 7 * 24 * 60 * 60
    assert document["financial_calls"]["place_order"] == 0


def test_contract_identity_document_exports_future_ibkr_fields() -> None:
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    contract = resolved_contract_from_contract_details(_future_contract_details("GC", 999001))
    row = ContractCacheRow(contract=contract, resolved_at=resolved_at, server_version=225)

    document = contract_identity_document(row)

    assert document["schema"] == "ibkr_contract_identity_v1"
    assert document["contract"]["conId"] == 999001
    assert document["contract"]["secType"] == "FUT"
    assert document["contract"]["lastTradeDateOrContractMonth"] == "202612"
    assert document["contract"]["multiplier"] == "100"
    assert document["contract"]["minTick"] == "0.1"
    assert document["contract"]["underConId"] == 12345
    assert document["contract_hash"] == contract_hash(contract)
    assert document["cache_ttl_seconds"] == 24 * 60 * 60
    assert document["financial_calls"]["global_cancel"] == 0


def test_contract_identity_document_exports_optional_future_reference_metadata() -> None:
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    contract = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
        min_tick=Decimal("0.1"),
        expiry="202612",
        market_rule_ids="332",
        last_trade_date_or_contract_month="202612",
        real_expiration_date="20261228",
        last_trade_time="12:30:00",
        under_con_id=12345,
        first_notice_day=datetime(2026, 11, 30, tzinfo=UTC).date(),
        last_trade_day=datetime(2026, 12, 28, tzinfo=UTC).date(),
        delivery_type="PHYSICAL",
        settlement_type="DELIVERABLE",
        contract_size_unit="TROY_OUNCE",
        roll_group="COMEX_GC",
        time_zone_id="US/Eastern",
        trading_hours="20260720:1800-1700",
        liquid_hours="20260720:1800-1700",
    )
    row = ContractCacheRow(contract=contract, resolved_at=resolved_at, server_version=225)

    payload = canonical_contract_payload(contract)
    document = contract_identity_document(row)

    assert payload["deliveryType"] == "PHYSICAL"
    assert payload["settlementType"] == "DELIVERABLE"
    assert payload["contractSizeUnit"] == "TROY_OUNCE"
    assert payload["rollGroup"] == "COMEX_GC"
    assert document["contract"]["firstNoticeDay"] == "2026-11-30"
    assert document["contract"]["lastTradeDay"] == "2026-12-28"
    assert document["contract_hash"] == contract_hash(contract)


def test_contract_resolution_artifacts_do_not_cache_ambiguous_payload_evaluation() -> None:
    request = ContractResolutionRequest(
        symbol="AIR",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="EUR",
        exchange="SMART",
    )
    evaluation = evaluate_contract_details_payloads(
        request,
        [
            _stock_contract_details("AIR", 111, "SBF", currency="EUR"),
            _stock_contract_details("AIR", 222, "AEB", currency="EUR"),
        ],
    )

    artifacts = build_contract_resolution_artifacts(
        request,
        evaluation,
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=225,
    )

    assert artifacts.resolved is False
    assert artifacts.cache_row is None
    assert artifacts.audit_record["resolution_status"] == "AMBIGUOUS_BLOCKED"
    assert artifacts.audit_record["selected_conId"] is None
    assert artifacts.audit_record["contract_hash"] is None


def test_persist_contract_resolution_artifacts_writes_resolved_audit_and_cache(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    evaluation = evaluate_contract_details_payloads(request, [_stock_contract_details("AAPL", 265598, "NASDAQ")])
    artifacts = build_contract_resolution_artifacts(
        request,
        evaluation,
        resolved_at=resolved_at,
        server_version=225,
    )

    report = persist_contract_resolution_artifacts(layout, artifacts)
    audit_record = json.loads(layout.requests_jsonl.read_text(encoding="utf-8"))
    rows = read_contract_cache_rows(layout)

    assert report["status"] == "GO"
    assert report["resolution_status"] == "RESOLVED"
    assert report["cache_written"] is True
    assert report["financial_calls"]["place_order"] == 0
    assert audit_record["selected_conId"] == 265598
    assert len(rows) == 1
    assert rows[0].contract.con_id == 265598
    assert rows[0].resolved_at == resolved_at
    assert not layout.errors_jsonl.exists()


def test_persist_contract_resolution_artifacts_routes_ambiguous_to_errors_without_cache(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    request = ContractResolutionRequest(
        symbol="AIR",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="EUR",
        exchange="SMART",
    )
    evaluation = evaluate_contract_details_payloads(
        request,
        [
            _stock_contract_details("AIR", 111, "SBF", currency="EUR"),
            _stock_contract_details("AIR", 222, "AEB", currency="EUR"),
        ],
    )
    artifacts = build_contract_resolution_artifacts(
        request,
        evaluation,
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=225,
    )

    report = persist_contract_resolution_artifacts(layout, artifacts)
    error_record = json.loads(layout.errors_jsonl.read_text(encoding="utf-8"))

    assert report["status"] == "GO"
    assert report["resolution_status"] == "AMBIGUOUS_BLOCKED"
    assert report["cache_written"] is False
    assert error_record["resolution_status"] == "AMBIGUOUS_BLOCKED"
    assert error_record["contract_hash"] is None
    assert read_contract_cache_rows(layout) == []
    assert not layout.requests_jsonl.exists()


def test_contract_cache_validation_accepts_valid_audit_jsonl(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )
    resolved_evaluation = evaluate_contract_details_payloads(
        request,
        [_stock_contract_details("AAPL", 265598, "NASDAQ")],
    )
    resolved_artifacts = build_contract_resolution_artifacts(
        request,
        resolved_evaluation,
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=225,
    )
    persist_contract_resolution_artifacts(layout, resolved_artifacts)
    missing_request = ContractResolutionRequest(
        symbol="MISSING",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
    )
    missing_evaluation = evaluate_contract_details_payloads(missing_request, [])
    missing_artifacts = build_contract_resolution_artifacts(
        missing_request,
        missing_evaluation,
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=225,
    )
    persist_contract_resolution_artifacts(layout, missing_artifacts)

    report = validate_contract_cache(layout)

    assert report["status"] == "GO"
    assert report["audit_row_count"] == 2
    assert report["files"]["contract_requests_jsonl"]["row_count"] == 1
    assert report["files"]["contract_errors_jsonl"]["row_count"] == 1


def test_contract_cache_validation_blocks_resolved_record_in_error_audit(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    record = {
        "requested_symbol": "AAPL",
        "requested_security_type": "STK",
        "requested_currency": "USD",
        "requested_exchange": "SMART",
        "requested_primary_exchange": "NASDAQ",
        "returned_match_count": 1,
        "selected_conId": 265598,
        "resolution_status": "RESOLVED",
        "resolved_at": "2026-07-20T00:00:00+00:00",
        "server_version": 225,
        "contract_hash": "abc",
    }
    append_contract_error_audit(layout, record)

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert report["files"]["contract_errors_jsonl"]["error_count"] >= 1
    assert any("unexpected resolution_status RESOLVED" in error for error in report["errors"])


def test_contract_cache_validation_blocks_resolved_request_audit_without_hash(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    record = {
        "requested_symbol": "AAPL",
        "requested_security_type": "STK",
        "requested_currency": "USD",
        "requested_exchange": "SMART",
        "requested_primary_exchange": "NASDAQ",
        "returned_match_count": 1,
        "selected_conId": 265598,
        "resolution_status": "RESOLVED",
        "resolved_at": "2026-07-20T00:00:00+00:00",
        "server_version": 225,
        "contract_hash": None,
    }
    append_contract_request_audit(layout, record)

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert any("contract_hash is required" in error for error in report["errors"])


def test_contract_cache_validation_blocks_audit_with_negative_server_version(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    record = {
        "requested_symbol": "AAPL",
        "requested_security_type": "STK",
        "requested_currency": "USD",
        "requested_exchange": "SMART",
        "requested_primary_exchange": "NASDAQ",
        "returned_match_count": 1,
        "selected_conId": 265598,
        "resolution_status": "RESOLVED",
        "resolved_at": "2026-07-20T00:00:00+00:00",
        "server_version": -1,
        "contract_hash": "abc",
    }
    append_contract_request_audit(layout, record)

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert any("server_version must be null or a positive integer" in error for error in report["errors"])


def test_contract_cache_validation_blocks_non_canonical_audit_request_fields(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    record = {
        "requested_symbol": "aapl",
        "requested_security_type": "STK",
        "requested_currency": "USD",
        "requested_exchange": "SMART",
        "requested_primary_exchange": "NASDAQ",
        "returned_match_count": 0,
        "selected_conId": None,
        "resolution_status": "NOT_FOUND",
        "resolved_at": "2026-07-20T00:00:00+00:00",
        "server_version": 225,
        "contract_hash": None,
    }
    append_contract_error_audit(layout, record)

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert any("requested_symbol must be an uppercase IBKR code" in error for error in report["errors"])


def test_contract_cache_validation_blocks_audit_with_naive_resolved_at(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    record = {
        "requested_symbol": "AAPL",
        "requested_security_type": "STK",
        "requested_currency": "USD",
        "requested_exchange": "SMART",
        "requested_primary_exchange": "NASDAQ",
        "returned_match_count": 1,
        "selected_conId": 265598,
        "resolution_status": "RESOLVED",
        "resolved_at": "2026-07-20T00:00:00",
        "server_version": 225,
        "contract_hash": "abc",
    }
    append_contract_request_audit(layout, record)

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert any("resolved_at must be a timezone-aware ISO timestamp" in error for error in report["errors"])


def test_contract_cache_validation_blocks_resolved_audit_with_wrong_match_count(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    record = {
        "requested_symbol": "AAPL",
        "requested_security_type": "STK",
        "requested_currency": "USD",
        "requested_exchange": "SMART",
        "requested_primary_exchange": "NASDAQ",
        "returned_match_count": 2,
        "selected_conId": 265598,
        "resolution_status": "RESOLVED",
        "resolved_at": "2026-07-20T00:00:00+00:00",
        "server_version": 225,
        "contract_hash": "abc",
    }
    append_contract_request_audit(layout, record)

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert any("returned_match_count must be 1 for RESOLVED" in error for error in report["errors"])


def test_contract_cache_validation_blocks_not_found_with_nonzero_match_count(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    record = {
        "requested_symbol": "MISSING",
        "requested_security_type": "STK",
        "requested_currency": "USD",
        "requested_exchange": "SMART",
        "requested_primary_exchange": None,
        "returned_match_count": 1,
        "selected_conId": None,
        "resolution_status": "NOT_FOUND",
        "resolved_at": "2026-07-20T00:00:00+00:00",
        "server_version": 225,
        "contract_hash": None,
    }
    append_contract_error_audit(layout, record)

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert any("returned_match_count must be 0 for NOT_FOUND" in error for error in report["errors"])


def test_contract_cache_validation_blocks_ambiguous_with_single_match_count(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    record = {
        "requested_symbol": "AIR",
        "requested_security_type": "STK",
        "requested_currency": "EUR",
        "requested_exchange": "SMART",
        "requested_primary_exchange": None,
        "returned_match_count": 1,
        "selected_conId": None,
        "resolution_status": "AMBIGUOUS_BLOCKED",
        "resolved_at": "2026-07-20T00:00:00+00:00",
        "server_version": 225,
        "contract_hash": None,
    }
    append_contract_error_audit(layout, record)

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert any("returned_match_count must be greater than 1 for AMBIGUOUS_BLOCKED" in error for error in report["errors"])


def test_upsert_contract_cache_row_replaces_existing_con_id(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    first_details = _stock_contract_details("AAPL", 265598, "NASDAQ")
    second_details = _stock_contract_details("AAPL", 265598, "NASDAQ")
    second_details.longName = "Apple Inc. Updated"
    first_row = ContractCacheRow(
        contract=resolved_contract_from_contract_details(first_details),
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=224,
    )
    second_row = ContractCacheRow(
        contract=resolved_contract_from_contract_details(second_details),
        resolved_at=datetime(2026, 7, 21, tzinfo=UTC),
        server_version=225,
    )

    upsert_contract_cache_row(layout, first_row)
    report = upsert_contract_cache_row(layout, second_row)
    rows = read_contract_cache_rows(layout)

    assert report["status"] == "GO"
    assert len(rows) == 1
    assert rows[0].contract.con_id == 265598
    assert rows[0].contract.long_name == "Apple Inc. Updated"
    assert rows[0].server_version == 225


def test_contract_cache_write_blocks_negative_server_version(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    row = ContractCacheRow(
        contract=resolved_contract_from_contract_details(_stock_contract_details("AAPL", 265598, "NASDAQ")),
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=-1,
    )

    with pytest.raises(ValueError, match="server_version must be null or a positive integer"):
        write_contract_cache_rows(layout, [row])


def test_contract_cache_write_blocks_naive_resolved_at(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    row = ContractCacheRow(
        contract=resolved_contract_from_contract_details(_stock_contract_details("AAPL", 265598, "NASDAQ")),
        resolved_at=datetime(2026, 7, 20),
        server_version=225,
    )

    with pytest.raises(ValueError, match="resolved_at must be timezone-aware"):
        write_contract_cache_rows(layout, [row])


def test_contract_cache_write_stocks_parquet(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    row = ContractCacheRow(
        contract=ResolvedContract(
            con_id=265598,
            symbol="AAPL",
            local_symbol="AAPL",
            security_type=IbkrSecurityType.STK,
            exchange="NASDAQ",
            primary_exchange="NASDAQ",
            currency="USD",
            trading_class="NMS",
            min_tick=Decimal("0.01"),
            valid_exchanges="SMART,NASDAQ",
            market_rule_ids="26",
            long_name="Apple Inc.",
            industry="Technology",
            category="Computers",
            subcategory="Computer Hardware",
            time_zone_id="US/Eastern",
            trading_hours="20260720:0930-1600",
            liquid_hours="20260720:0930-1600",
        ),
        resolved_at=resolved_at,
        server_version=225,
    )

    report = write_contract_cache_rows(layout, [row])
    records = read_contract_cache_records(layout.stocks_parquet)

    assert report["status"] == "GO"
    assert report["written"]["STK"]["row_count"] == 1
    assert report["financial_calls"]["place_order"] == 0
    assert layout.stocks_parquet.exists()
    assert records[0]["con_id"] == 265598
    assert records[0]["security_type"] == "STK"
    assert records[0]["contract_hash"] == contract_hash(row.contract)
    assert records[0]["resolved_at"] == resolved_at
    assert records[0]["server_version"] == 225


def test_contract_cache_write_futures_parquet(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    row = ContractCacheRow(
        contract=ResolvedContract(
            con_id=999001,
            symbol="GC",
            local_symbol="GCZ6",
            security_type=IbkrSecurityType.FUT,
            exchange="COMEX",
            currency="USD",
            trading_class="GC",
            multiplier=Decimal("100"),
            min_tick=Decimal("0.1"),
            expiry="202612",
            market_rule_ids="332",
            last_trade_date_or_contract_month="202612",
            real_expiration_date="20261228",
            last_trade_time="12:30:00",
            under_con_id=12345,
            time_zone_id="US/Eastern",
            trading_hours="20260720:1800-1700",
            liquid_hours="20260720:1800-1700",
        ),
        resolved_at=resolved_at,
        server_version=225,
    )

    report = write_contract_cache_rows(layout, [row])
    records = read_contract_cache_records(layout.futures_parquet)

    assert report["status"] == "GO"
    assert report["written"]["FUT"]["row_count"] == 1
    assert layout.futures_parquet.exists()
    assert records[0]["con_id"] == 999001
    assert records[0]["security_type"] == "FUT"
    assert records[0]["multiplier"] == 100.0
    assert records[0]["expiry"] == "202612"
    assert records[0]["contract_hash"] == contract_hash(row.contract)
    assert records[0]["resolved_at"] == resolved_at
    assert records[0]["server_version"] == 225


def test_contract_cache_validation_blocks_duplicate_con_id_across_files(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    stock_row = ContractCacheRow(
        contract=ResolvedContract(
            con_id=123456,
            symbol="AAPL",
            local_symbol="AAPL",
            security_type=IbkrSecurityType.STK,
            exchange="NASDAQ",
            primary_exchange="NASDAQ",
            currency="USD",
            trading_class="NMS",
            min_tick=Decimal("0.01"),
            valid_exchanges="SMART,NASDAQ",
            market_rule_ids="26",
            long_name="Apple Inc.",
            industry="Technology",
            category="Computers",
            subcategory="Computer Hardware",
            time_zone_id="US/Eastern",
            trading_hours="20260720:0930-1600",
            liquid_hours="20260720:0930-1600",
        ),
        resolved_at=resolved_at,
        server_version=225,
    )
    future_row = ContractCacheRow(
        contract=ResolvedContract(
            con_id=123456,
            symbol="GC",
            local_symbol="GCZ6",
            security_type=IbkrSecurityType.FUT,
            exchange="COMEX",
            currency="USD",
            trading_class="GC",
            multiplier=Decimal("100"),
            min_tick=Decimal("0.1"),
            expiry="202612",
            market_rule_ids="332",
            last_trade_date_or_contract_month="202612",
            real_expiration_date="20261228",
            last_trade_time="12:30:00",
            under_con_id=12345,
            time_zone_id="US/Eastern",
            trading_hours="20260720:1800-1700",
            liquid_hours="20260720:1800-1700",
        ),
        resolved_at=resolved_at,
        server_version=225,
    )
    write_contract_cache_rows(layout, [stock_row])
    write_contract_cache_rows(layout, [future_row])

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert report["row_count"] == 2
    assert "duplicate or invalid con_id" in report["errors"][0]


def test_contract_cache_rows_roundtrip_from_parquet_supports_cache_hit(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    stock_row = ContractCacheRow(
        contract=ResolvedContract(
            con_id=265598,
            symbol="AAPL",
            local_symbol="AAPL",
            security_type=IbkrSecurityType.STK,
            exchange="NASDAQ",
            primary_exchange="NASDAQ",
            currency="USD",
            trading_class="NMS",
            min_tick=Decimal("0.01"),
            valid_exchanges="SMART,NASDAQ",
            market_rule_ids="26",
            long_name="Apple Inc.",
            industry="Technology",
            category="Computers",
            subcategory="Computer Hardware",
            time_zone_id="US/Eastern",
            trading_hours="20260720:0930-1600",
            liquid_hours="20260720:0930-1600",
        ),
        resolved_at=resolved_at,
        server_version=225,
    )
    future_row = ContractCacheRow(
        contract=ResolvedContract(
            con_id=999001,
            symbol="GC",
            local_symbol="GCZ6",
            security_type=IbkrSecurityType.FUT,
            exchange="COMEX",
            currency="USD",
            trading_class="GC",
            multiplier=Decimal("100"),
            min_tick=Decimal("0.1"),
            expiry="202612",
            market_rule_ids="332",
            last_trade_date_or_contract_month="202612",
            real_expiration_date="20261228",
            last_trade_time="12:30:00",
            under_con_id=12345,
            time_zone_id="US/Eastern",
            trading_hours="20260720:1800-1700",
            liquid_hours="20260720:1800-1700",
        ),
        resolved_at=resolved_at,
        server_version=225,
    )
    write_contract_cache_rows(layout, [stock_row, future_row])

    rows = read_contract_cache_rows(layout)
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
    )
    hit = find_fresh_contract_cache_hit(rows, request, now=resolved_at + timedelta(days=1))

    assert len(rows) == 2
    assert hit is not None
    assert hit.contract.con_id == 265598
    assert hit.contract.min_tick == Decimal("0.01")
    assert hit.contract_hash == contract_hash(stock_row.contract)
    assert {row.contract.security_type for row in rows} == {IbkrSecurityType.STK, IbkrSecurityType.FUT}


def test_future_reference_metadata_roundtrips_through_parquet_cache(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    contract = ResolvedContract(
        con_id=999001,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
        min_tick=Decimal("0.1"),
        expiry="202612",
        market_rule_ids="332",
        last_trade_date_or_contract_month="202612",
        real_expiration_date="20261228",
        last_trade_time="12:30:00",
        under_con_id=12345,
        first_notice_day=datetime(2026, 11, 30, tzinfo=UTC).date(),
        last_trade_day=datetime(2026, 12, 28, tzinfo=UTC).date(),
        delivery_type="PHYSICAL",
        settlement_type="DELIVERABLE",
        contract_size_unit="TROY_OUNCE",
        roll_group="COMEX_GC",
        time_zone_id="US/Eastern",
        trading_hours="20260720:1800-1700",
        liquid_hours="20260720:1800-1700",
    )
    write_contract_cache_rows(
        layout,
        [ContractCacheRow(contract=contract, resolved_at=resolved_at, server_version=225)],
    )

    rows = read_contract_cache_rows(layout)

    assert len(rows) == 1
    assert rows[0].contract.first_notice_day.isoformat() == "2026-11-30"
    assert rows[0].contract.last_trade_day.isoformat() == "2026-12-28"
    assert rows[0].contract.delivery_type == "PHYSICAL"
    assert rows[0].contract.settlement_type == "DELIVERABLE"
    assert rows[0].contract.contract_size_unit == "TROY_OUNCE"
    assert rows[0].contract.roll_group == "COMEX_GC"
    assert rows[0].contract_hash == contract_hash(contract)


def test_contract_cache_validation_blocks_contract_hash_mismatch(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    row = ContractCacheRow(
        contract=resolved_contract_from_contract_details(_stock_contract_details("AAPL", 265598, "NASDAQ")),
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=225,
    )
    write_contract_cache_rows(layout, [row])
    records = read_contract_cache_records(layout.stocks_parquet)
    records[0]["contract_hash"] = "BAD_HASH"
    pq.write_table(pa.Table.from_pylist(records), layout.stocks_parquet)

    report = validate_contract_cache(layout)

    assert report["status"] == "NO_GO"
    assert report["files"]["stocks_parquet"]["error_count"] == 1
    assert "contract_hash mismatch" in report["errors"][0]


def test_contract_cache_write_blocks_duplicate_con_id(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    resolved_at = datetime(2026, 7, 20, tzinfo=UTC)
    contract = ResolvedContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type=IbkrSecurityType.STK,
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        min_tick=Decimal("0.01"),
        valid_exchanges="SMART,NASDAQ",
        market_rule_ids="26",
        long_name="Apple Inc.",
        industry="Technology",
        category="Computers",
        subcategory="Computer Hardware",
        time_zone_id="US/Eastern",
        trading_hours="20260720:0930-1600",
        liquid_hours="20260720:0930-1600",
    )
    rows = [
        ContractCacheRow(contract=contract, resolved_at=resolved_at),
        ContractCacheRow(contract=contract, resolved_at=resolved_at),
    ]

    with pytest.raises(ValueError, match="duplicate conId"):
        write_contract_cache_rows(layout, rows)

    assert read_contract_cache_records(layout.stocks_parquet) == []


def test_contract_cache_write_blocks_missing_required_phase2_fields(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    row = ContractCacheRow(
        contract=ResolvedContract(
            con_id=265598,
            symbol="AAPL",
            local_symbol="AAPL",
            security_type=IbkrSecurityType.STK,
            exchange="NASDAQ",
            currency="USD",
            trading_class="NMS",
            min_tick=Decimal("0.01"),
            valid_exchanges="SMART,NASDAQ",
            market_rule_ids="26",
            long_name="Apple Inc.",
            industry="Technology",
            category="Computers",
            subcategory="Computer Hardware",
            time_zone_id="US/Eastern",
            trading_hours="20260720:0930-1600",
            liquid_hours="20260720:0930-1600",
        ),
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="primary_exchange"):
        write_contract_cache_rows(layout, [row])

    assert read_contract_cache_records(layout.stocks_parquet) == []


def test_typed_stock_and_future_requests_map_to_resolution_requests() -> None:
    stock = StockContractRequest(
        symbol="AAPL",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    ).to_resolution_request()
    future = FutureContractRequest(
        symbol="GC",
        currency="USD",
        exchange="COMEX",
    ).to_resolution_request()

    assert stock.security_type == IbkrSecurityType.STK
    assert stock.primary_exchange == "NASDAQ"
    assert future.security_type == IbkrSecurityType.FUT
    assert future.expiry is None


def test_contract_request_hash_is_deterministic() -> None:
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )

    assert contract_request_hash(request) == contract_request_hash(request)
    assert len(contract_request_hash(request)) == 64


def test_contract_request_id_allocator_is_unique() -> None:
    allocator = ContractRequestIdAllocator(start=500)

    assert [allocator.next_id() for _ in range(4)] == [500, 501, 502, 503]


def test_live_contract_resolver_resolves_stock_with_fake_contract_details(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )
    service = ReadOnlyIbkrConnectionService(
        _resolver_settings(tmp_path),
        app_factory=lambda state: ContractDetailsFakeApp(
            state,
            [_stock_contract_details("AAPL", 265598, "NASDAQ")],
        ),
    )

    result = LiveContractResolver(service, layout, timeout_seconds=0.05).resolve(request).as_dict()

    assert result["status"] == "RESOLVED"
    assert result["source"] == "ibkr_tws_paper"
    assert result["read_only_ibkr_calls"]["req_contract_details"] == 1
    assert result["read_only_ibkr_calls"]["req_mkt_data"] == 0
    assert result["read_only_ibkr_calls"]["req_historical_data"] == 0
    assert result["resolved_contract"]["contract"]["conId"] == 265598
    assert result["financial_calls"]["place_order"] == 0
    assert validate_contract_cache(layout)["status"] == "GO"
    assert read_contract_cache_rows(layout)[0].contract.con_id == 265598


def test_live_contract_resolver_blocks_ambiguous_future_chain_without_selecting_first(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    request = ContractResolutionRequest(
        symbol="GC",
        asset_class=AssetClass.COMMODITY_FUTURE,
        security_type=IbkrSecurityType.FUT,
        currency="USD",
        exchange="COMEX",
    )
    service = ReadOnlyIbkrConnectionService(
        _resolver_settings(tmp_path),
        app_factory=lambda state: ContractDetailsFakeApp(
            state,
            [
                _future_contract_details("GC", 900001),
                _future_contract_details("GC", 900002),
            ],
        ),
    )

    result = LiveContractResolver(service, layout, timeout_seconds=0.05).resolve(request).as_dict()

    assert result["status"] == "AMBIGUOUS_BLOCKED"
    assert result["returned_match_count"] == 2
    assert result["resolved_contract"] is None
    assert result["persistence"]["cache_written"] is False
    assert result["financial_calls"]["global_cancel"] == 0


def test_live_contract_resolver_reports_callback_timeout(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
    )
    service = ReadOnlyIbkrConnectionService(
        _resolver_settings(tmp_path),
        app_factory=lambda state: ContractDetailsFakeApp(
            state,
            [_stock_contract_details("AAPL", 265598, "NASDAQ")],
            send_end=False,
        ),
    )

    result = LiveContractResolver(service, layout, timeout_seconds=0.001).resolve(request).as_dict()

    assert result["status"] == "CALLBACK_TIMEOUT"
    assert result["resolved_contract"] is None
    assert result["read_only_ibkr_calls"]["req_contract_details"] == 1
    assert result["financial_calls"]["cancel_order"] == 0


def test_live_contract_resolver_uses_fresh_cache_before_broker_request(tmp_path) -> None:
    layout = ContractCacheLayout.from_project_root(tmp_path)
    request = ContractResolutionRequest(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        security_type=IbkrSecurityType.STK,
        currency="USD",
        exchange="NASDAQ",
        primary_exchange="NASDAQ",
    )
    row = ContractCacheRow(
        contract=resolved_contract_from_contract_details(_stock_contract_details("AAPL", 265598, "NASDAQ")),
        resolved_at=datetime.now(UTC),
        server_version=225,
    )
    write_contract_cache_rows(layout, [row])
    service = ReadOnlyIbkrConnectionService(
        _resolver_settings(tmp_path),
        app_factory=lambda state: ContractDetailsFakeApp(state, []),
    )

    result = LiveContractResolver(service, layout, timeout_seconds=0.05).resolve(request).as_dict()

    assert result["status"] == "RESOLVED"
    assert result["source"] == "local_contract_cache"
    assert result["cache_hit"] is True
    assert result["request_id"] is None
    assert result["read_only_ibkr_calls"]["req_contract_details"] == 0


class ContractDetailsFakeApp:
    def __init__(
        self,
        state: CallbackState,
        details: list[ContractDetails],
        *,
        send_end: bool = True,
    ) -> None:
        self.state = state
        self.details = details
        self.send_end = send_end
        self.connected = False

    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.state.record_closed()

    def isConnected(self) -> bool:  # noqa: N802
        return self.connected

    def serverVersion(self) -> int:  # noqa: N802
        return 225

    def twsConnectionTime(self) -> str:  # noqa: N802
        return "20260720 00:00:00"

    def run(self) -> None:
        self.state.record_next_valid_id(1001)
        self.reqCurrentTime()
        self.reqManagedAccts()
        while self.connected:
            time.sleep(0.001)

    def reqCurrentTime(self) -> None:  # noqa: N802
        self.state.record_current_time(1784592000)

    def reqManagedAccts(self) -> None:  # noqa: N802
        self.state.record_managed_accounts("DU" + "1234567")

    def reqContractDetails(self, req_id: int, contract: Contract) -> None:  # noqa: N802
        for detail in self.details:
            self.state.record_contract_details(req_id, detail)
        if self.send_end:
            self.state.record_contract_details_end(req_id)

    def reqMatchingSymbols(self, req_id: int, pattern: str) -> None:  # noqa: N802
        self.state.record_symbol_samples(req_id, [])

    def reqMarketRule(self, market_rule_id: int) -> None:  # noqa: N802
        self.state.record_market_rule(market_rule_id, [])


def _resolver_settings(tmp_path: Path) -> IbkrSettings:
    return IbkrSettings(
        output_dir=tmp_path,
        connect_timeout_seconds=0.05,
        request_timeout_seconds=0.05,
        heartbeat_interval_seconds=0.01,
        stale_after_seconds=45.0,
    )


def _write_phase1_drill_artifact(
    tmp_path: Path,
    *,
    file_name: str = "phase1-disconnect-drill-20260720-000000.json",
    port: int = 7497,
    client_id: object = 17,
    financial_calls: dict[str, object] | None = None,
    observed_statuses: list[dict[str, str]] | None = None,
) -> Path:
    artifact_dir = tmp_path / "output" / "ibkr"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / file_name
    artifact = {
        "schema": "ibkr_forced_disconnect_drill_v1",
        "generated_at": "2026-07-20T00:00:00Z",
        "status": "GO",
        "host": "127.0.0.1",
        "port": port,
        "client_id": client_id,
        "seconds": 180.0,
        "poll_seconds": 2.0,
        "disconnect_observed": True,
        "reconnect_successful": True,
        "failure_reason": None,
        "observed_statuses": observed_statuses
        or [
            {"phase": "start", "status": "HEALTHY"},
            {"phase": "heartbeat", "status": "DISCONNECTED"},
            {"phase": "reconnect", "status": "HEALTHY"},
        ],
        "financial_calls": financial_calls
        or {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path


def _write_required_phase1_files(tmp_path: Path) -> None:
    for file_name in (
        "ibkr_tws_probe.py",
        "requirements.lock.txt",
        "main.py",
        "src/stocks/application/config.py",
        "src/stocks/application/context.py",
        "src/stocks/application/phase_gates.py",
        "src/stocks/application/lifecycle.py",
        "src/stocks/ibkr/connection.py",
        "src/stocks/ibkr/client.py",
        "src/stocks/ibkr/callbacks.py",
        "src/stocks/ibkr/errors.py",
        "src/stocks/ibkr/health.py",
    ):
        path = tmp_path / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{file_name}\n", encoding="utf-8")


def _write_phase1_freeze_report(
    tmp_path: Path,
    artifact_path: Path,
    *,
    artifact_hash: str | None = None,
) -> None:
    effective_hash = artifact_hash or hashlib.sha256(artifact_path.read_bytes()).hexdigest().upper()
    phase0_hash_lines = _freeze_hash_lines(
        tmp_path,
        (
            "ibkr_tws_probe.py",
            "requirements.lock.txt",
        ),
    )
    phase1_hash_lines = _freeze_hash_lines(
        tmp_path,
        (
            "main.py",
            "src/stocks/application/config.py",
            "src/stocks/application/context.py",
            "src/stocks/application/phase_gates.py",
            "src/stocks/application/lifecycle.py",
            "src/stocks/ibkr/connection.py",
            "src/stocks/ibkr/client.py",
            "src/stocks/ibkr/callbacks.py",
            "src/stocks/ibkr/errors.py",
            "src/stocks/ibkr/health.py",
        ),
    )
    freeze_report = f"""# Phase 1 Freeze Report

Status:

```text
IBKR_PHASE_1_READ_ONLY_CONNECTION_SERVICE_GO
PHASE1_CONNECTION_SERVICE_FROZEN_GO
```

Verified artifact:

```text
{artifact_path}
```

Artifact SHA256:

```text
{effective_hash}
```

Evidence:

```text
schema                  ibkr_forced_disconnect_drill_v1
status                  GO
disconnect_observed     True
reconnect_successful    True
place_order             0
cancel_order            0
global_cancel           0
```

Frozen Phase 0 hashes:

```text
{phase0_hash_lines}
```

Frozen Phase 1 hashes:

```text
{phase1_hash_lines}
```
"""
    (tmp_path / "PHASE1_FREEZE_REPORT.md").write_text(freeze_report, encoding="utf-8")


def _freeze_hash_lines(tmp_path: Path, file_names: tuple[str, ...]) -> str:
    lines = []
    for file_name in file_names:
        digest = hashlib.sha256((tmp_path / file_name).read_bytes()).hexdigest().upper()
        lines.append(f"{file_name:<34} {digest}")
    return "\n".join(lines)


def _stock_contract_details(
    symbol: str,
    con_id: int,
    primary_exchange: str,
    *,
    currency: str = "USD",
) -> ContractDetails:
    native_contract = Contract()
    native_contract.conId = con_id
    native_contract.symbol = symbol
    native_contract.localSymbol = symbol
    native_contract.secType = "STK"
    native_contract.exchange = primary_exchange
    native_contract.primaryExchange = primary_exchange
    native_contract.currency = currency
    native_contract.tradingClass = symbol
    details = ContractDetails()
    details.contract = native_contract
    details.minTick = 0.01
    details.validExchanges = f"SMART,{primary_exchange}"
    details.marketRuleIds = "26"
    details.longName = f"{symbol} Test Instrument"
    details.industry = "Technology"
    details.category = "Computers"
    details.subcategory = "Computer Hardware"
    details.timeZoneId = "US/Eastern"
    details.tradingHours = "20260720:0930-1600"
    details.liquidHours = "20260720:0930-1600"
    return details


def _future_contract_details(symbol: str, con_id: int) -> ContractDetails:
    native_contract = Contract()
    native_contract.conId = con_id
    native_contract.symbol = symbol
    native_contract.localSymbol = f"{symbol}Z6"
    native_contract.secType = "FUT"
    native_contract.exchange = "COMEX"
    native_contract.currency = "USD"
    native_contract.tradingClass = symbol
    native_contract.multiplier = "100"
    native_contract.lastTradeDateOrContractMonth = "202612"
    details = ContractDetails()
    details.contract = native_contract
    details.minTick = 0.1
    details.marketRuleIds = "332"
    details.realExpirationDate = "20261228"
    details.lastTradeTime = "12:30:00"
    details.underConId = 12345
    details.timeZoneId = "US/Eastern"
    details.tradingHours = "20260720:1800-1700"
    details.liquidHours = "20260720:1800-1700"
    return details
