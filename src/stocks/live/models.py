from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class LiveCanaryConfig:
    host: str
    port: int
    writer_client_id: int
    recon_client_id: int
    quote_client_id: int
    account_fingerprint_key: str
    approved_account_fingerprint: str
    manual_activation_phrase: str
    writer_enabled: bool
    max_order_eur: Decimal
    max_total_exposure_eur: Decimal
    max_risk_eur: Decimal
    max_open_positions: int
    max_new_orders_per_day: int
    approval_ttl_seconds: int
    callback_timeout_seconds: float
    fractional_shares_enabled: bool
    execution_authority: str = "LIVE_LEVEL_ONE"
    maximum_quantity: Decimal = Decimal("100")
    canary_risk_fraction: Decimal = Decimal("0.005")
    maximum_stock_weight: Decimal = Decimal("0.15")
    maximum_pooled_vehicle_weight: Decimal = Decimal("0.20")
    maximum_portfolio_heat_fraction: Decimal = Decimal("0.005")
    minimum_economic_notional_eur: Decimal = Decimal("5")
    maximum_cost_to_expected_edge_ratio: Decimal = Decimal("0.50")
    policy_version: str = "P0.1_WHOLE_SHARE_CANARY_V1"

    def safe_dict(self) -> dict[str, Any]:
        return {
            "host_local_only": self.host in {"127.0.0.1", "localhost"},
            "port_class": "LIVE" if self.port in {7496, 4001} else "INVALID",
            "writer_client_id_nonzero": self.writer_client_id != 0,
            "observer_client_id_nonzero": self.recon_client_id != 0,
            "client_ids_unique": self.writer_client_id != self.recon_client_id,
            "quote_client_id_nonzero": self.quote_client_id != 0,
            "all_client_ids_unique": len(
                {
                    self.writer_client_id,
                    self.recon_client_id,
                    self.quote_client_id,
                }
            )
            == 3,
            "fingerprint_key_configured": bool(self.account_fingerprint_key),
            "approved_account_fingerprint_configured": bool(
                self.approved_account_fingerprint
            ),
            "manual_activation_phrase_configured": bool(
                self.manual_activation_phrase
            ),
            "writer_enabled": self.writer_enabled,
            "max_order_eur": str(self.max_order_eur),
            "max_total_exposure_eur": str(self.max_total_exposure_eur),
            "max_risk_eur": str(self.max_risk_eur),
            "max_open_positions": self.max_open_positions,
            "max_new_orders_per_day": self.max_new_orders_per_day,
            "approval_ttl_seconds": self.approval_ttl_seconds,
            "callback_timeout_seconds": self.callback_timeout_seconds,
            "fractional_shares_enabled": self.fractional_shares_enabled,
            "execution_authority": self.execution_authority,
            "maximum_quantity": str(self.maximum_quantity),
            "canary_risk_fraction": str(self.canary_risk_fraction),
            "maximum_stock_weight": str(self.maximum_stock_weight),
            "maximum_pooled_vehicle_weight": str(
                self.maximum_pooled_vehicle_weight
            ),
            "maximum_portfolio_heat_fraction": str(
                self.maximum_portfolio_heat_fraction
            ),
            "minimum_economic_notional_eur": str(
                self.minimum_economic_notional_eur
            ),
            "maximum_cost_to_expected_edge_ratio": str(
                self.maximum_cost_to_expected_edge_ratio
            ),
            "policy_version": self.policy_version,
            "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
            "notional_cap_role": "SECONDARY_EMERGENCY_BACKSTOP",
        }


@dataclass(frozen=True)
class ManualLiveBracketIntent:
    intent_id: str
    economic_order_key: str
    created_at: str
    expires_at: str
    account_fingerprint: str
    con_id: int
    symbol: str
    security_type: str
    currency: str
    exchange: str
    quantity: Decimal
    entry_limit_price: Decimal
    stop_price: Decimal
    take_profit_price: Decimal
    fx_rate_to_eur: Decimal
    estimated_notional_eur: Decimal
    maximum_planned_loss_eur: Decimal
    session_date: str
    operator_reason: str
    contract_hash: str
    strategy_id: str | None = None
    target_id: str | None = None
    asset_class: str = "STOCK"
    desired_qty: Decimal = Decimal("0")
    normal_allowed_qty: Decimal = Decimal("0")
    canary_qty: Decimal = Decimal("0")
    risk_per_share_eur: Decimal = Decimal("0")
    planned_total_risk_eur: Decimal = Decimal("0")
    portfolio_weight: Decimal = Decimal("0")
    cash_before_eur: Decimal = Decimal("0")
    cash_after_eur: Decimal = Decimal("0")
    estimated_total_cost_eur: Decimal = Decimal("0")
    expected_net_opportunity_eur: Decimal = Decimal("0")
    canary_notional_hard_cap_eur: Decimal = Decimal("0")
    capital_level: int = 1
    sizing_reason: str = ""
    downscaled_for_canary: bool = False
    fractional_allowed: bool = False

    def jsonable(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
