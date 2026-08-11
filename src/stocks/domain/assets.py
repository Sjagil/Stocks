from __future__ import annotations

from enum import Enum


class AssetClass(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    BOND_ETF = "bond_etf"
    COMMODITY_ETF = "commodity_etf"
    COMMODITY_FUTURE = "commodity_future"
    CASH = "cash"


class IbkrSecurityType(str, Enum):
    STK = "STK"
    FUT = "FUT"


def ibkr_security_type_for(asset_class: AssetClass) -> IbkrSecurityType:
    if asset_class in {
        AssetClass.STOCK,
        AssetClass.ETF,
        AssetClass.BOND_ETF,
        AssetClass.COMMODITY_ETF,
    }:
        return IbkrSecurityType.STK
    if asset_class == AssetClass.COMMODITY_FUTURE:
        return IbkrSecurityType.FUT
    raise ValueError(f"{asset_class.value} is not an IBKR-tradable security class")
