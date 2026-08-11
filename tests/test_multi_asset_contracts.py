from __future__ import annotations

from decimal import Decimal

import pytest

from stocks.domain.assets import AssetClass, IbkrSecurityType, ibkr_security_type_for
from stocks.ibkr.contracts import (
    ContractQuery,
    ContractResolverNotEnabled,
    OfflineContractResolver,
    ResolvedContract,
)
from stocks.portfolio.sleeves import DEFAULT_SLEEVES, SleeveName, sleeve_by_name


def test_ibkr_security_type_mapping() -> None:
    assert ibkr_security_type_for(AssetClass.STOCK) == IbkrSecurityType.STK
    assert ibkr_security_type_for(AssetClass.ETF) == IbkrSecurityType.STK
    assert ibkr_security_type_for(AssetClass.COMMODITY_ETF) == IbkrSecurityType.STK
    assert ibkr_security_type_for(AssetClass.COMMODITY_FUTURE) == IbkrSecurityType.FUT


def test_cash_is_not_contract_query() -> None:
    query = ContractQuery(symbol="CASH", asset_class=AssetClass.CASH, currency="EUR")

    with pytest.raises(ValueError, match="cash"):
        query.validate()


def test_futures_query_requires_expiry() -> None:
    query = ContractQuery(symbol="GC", asset_class=AssetClass.COMMODITY_FUTURE, currency="USD")

    with pytest.raises(ValueError, match="expiry"):
        query.validate()


def test_contract_query_requires_canonical_symbol_and_exchange() -> None:
    query = ContractQuery(symbol="spy", asset_class=AssetClass.ETF, currency="USD", exchange=" SMART")

    with pytest.raises(ValueError, match="symbol must be an uppercase IBKR code"):
        query.validate()


def test_resolved_future_requires_multiplier_and_expiry() -> None:
    contract = ResolvedContract(
        con_id=123,
        symbol="GC",
        local_symbol="GCZ6",
        security_type=IbkrSecurityType.FUT,
        exchange="COMEX",
        currency="USD",
        trading_class="GC",
        multiplier=Decimal("100"),
        expiry="202612",
    )

    contract.validate_for_storage()


def test_offline_resolver_blocks_live_phase2_requests() -> None:
    resolver = OfflineContractResolver()
    query = ContractQuery(symbol="SPY", asset_class=AssetClass.ETF, currency="USD")

    with pytest.raises(ContractResolverNotEnabled):
        resolver.qualify(query)


def test_default_sleeves_cover_initial_control_plane() -> None:
    names = {sleeve.name for sleeve in DEFAULT_SLEEVES}

    assert SleeveName.EQUITY_MOMENTUM in names
    assert SleeveName.ETF_CORE_ROTATION in names
    assert SleeveName.COMMODITY_TREND in names
    assert SleeveName.DEFENSIVE_CASH in names
    assert sleeve_by_name(SleeveName.COMMODITY_CARRY).default_enabled is False
