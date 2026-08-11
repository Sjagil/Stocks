from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from stocks.domain.assets import AssetClass


class SleeveName(str, Enum):
    EQUITY_MOMENTUM = "equity_momentum"
    ETF_CORE_ROTATION = "etf_core_rotation"
    BOND_DURATION = "bond_duration"
    COMMODITY_TREND = "commodity_trend"
    COMMODITY_CARRY = "commodity_carry"
    DEFENSIVE_CASH = "defensive_cash"
    MEAN_REVERSION = "mean_reversion"


@dataclass(frozen=True)
class SleeveDefinition:
    name: SleeveName
    allowed_asset_classes: tuple[AssetClass, ...]
    default_enabled: bool
    requires_point_in_time_data: bool
    allows_short: bool = False


DEFAULT_SLEEVES: tuple[SleeveDefinition, ...] = (
    SleeveDefinition(
        name=SleeveName.EQUITY_MOMENTUM,
        allowed_asset_classes=(AssetClass.STOCK,),
        default_enabled=True,
        requires_point_in_time_data=True,
    ),
    SleeveDefinition(
        name=SleeveName.ETF_CORE_ROTATION,
        allowed_asset_classes=(
            AssetClass.ETF,
            AssetClass.BOND_ETF,
            AssetClass.COMMODITY_ETF,
            AssetClass.CASH,
        ),
        default_enabled=True,
        requires_point_in_time_data=False,
    ),
    SleeveDefinition(
        name=SleeveName.BOND_DURATION,
        allowed_asset_classes=(AssetClass.BOND_ETF,),
        default_enabled=True,
        requires_point_in_time_data=False,
    ),
    SleeveDefinition(
        name=SleeveName.COMMODITY_TREND,
        allowed_asset_classes=(AssetClass.COMMODITY_ETF, AssetClass.COMMODITY_FUTURE),
        default_enabled=True,
        requires_point_in_time_data=False,
    ),
    SleeveDefinition(
        name=SleeveName.COMMODITY_CARRY,
        allowed_asset_classes=(AssetClass.COMMODITY_FUTURE,),
        default_enabled=False,
        requires_point_in_time_data=True,
    ),
    SleeveDefinition(
        name=SleeveName.DEFENSIVE_CASH,
        allowed_asset_classes=(AssetClass.CASH, AssetClass.BOND_ETF, AssetClass.COMMODITY_ETF),
        default_enabled=True,
        requires_point_in_time_data=False,
    ),
    SleeveDefinition(
        name=SleeveName.MEAN_REVERSION,
        allowed_asset_classes=(AssetClass.ETF,),
        default_enabled=False,
        requires_point_in_time_data=False,
    ),
)


def sleeve_by_name(name: SleeveName) -> SleeveDefinition:
    for sleeve in DEFAULT_SLEEVES:
        if sleeve.name == name:
            return sleeve
    raise KeyError(name.value)
