from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

from dotenv import dotenv_values

from stocks.capital.canary import (
    default_level_one_canary_policy,
    load_level_one_canary_policy,
)
from stocks.live.models import LiveCanaryConfig
from stocks.capital.service import capital_level_limits


LIVE_PORTS = {7496, 4001}
LIVE_LEVEL_TWO = "LIVE_LEVEL_TWO"


def load_live_canary_config(
    project_root: Path,
    env_file: str | Path = ".env.ibkr.live",
) -> tuple[LiveCanaryConfig | None, list[str]]:
    path = Path(env_file)
    if not path.is_absolute():
        path = project_root / path
    errors: list[str] = []
    if path.name != ".env.ibkr.live" or not path.exists():
        return None, ["DEDICATED_LIVE_ENV_REQUIRED"]
    values = {
        key: str(value)
        for key, value in dotenv_values(path).items()
        if value is not None
    }
    try:
        policy = load_level_one_canary_policy(project_root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        policy = default_level_one_canary_policy()

    def get(name: str, default: str = "") -> str:
        return values.get(name, default).strip()

    host = get("IBKR_HOST")
    port = _int(get("IBKR_PORT"), -1)
    writer_id = _int(get("IBKR_CLIENT_ID"), -1)
    recon_id = _int(get("IBKR_RECON_CLIENT_ID"), -1)
    quote_id = _int(get("IBKR_QUOTE_CLIENT_ID"), -1)
    writer_enabled = (
        get("IBKR_ENVIRONMENT") == "LIVE"
        and not _bool(get("IBKR_READ_ONLY"), True)
        and get("IBKR_ORDER_AUTHORITY") == "CANARY"
        and _bool(get("IBKR_ALLOW_ORDER_TRANSMISSION"), False)
        and _bool(get("IBKR_LIVE_TRADING_ENABLED"), False)
    )
    max_order = _decimal(get("IBKR_MAX_ORDER_EUR"), Decimal("999999"))
    max_total = _decimal(
        get("IBKR_MAX_TOTAL_EXPOSURE_EUR"), Decimal("999999")
    )
    max_risk = _decimal(get("IBKR_MAX_RISK_EUR"), Decimal("999999"))
    max_positions = _int(get("IBKR_MAX_OPEN_POSITIONS"), 999)
    max_daily = _int(get("IBKR_MAX_NEW_ORDERS_PER_DAY"), 999)
    fractional_shares_enabled = _bool(
        get("IBKR_ALLOW_FRACTIONAL_SHARES"), False
    )
    ttl = _int(get("IBKR_LIVE_APPROVAL_TTL_SECONDS"), 300)
    callback_timeout = _float(
        get("IBKR_LIVE_CALLBACK_TIMEOUT_SECONDS"), 15.0
    )
    forbidden_products_disabled = all(
        not _bool(get(name), True)
        for name in (
            "IBKR_ALLOW_FUTURES",
            "IBKR_ALLOW_SHORTS",
            "IBKR_ALLOW_MARGIN",
            "IBKR_ALLOW_OPTIONS",
            "IBKR_ALLOW_FOREX_SPECULATION",
        )
    )
    if host not in {"127.0.0.1", "localhost"}:
        errors.append("LIVE_HOST_LOCAL_ONLY_REQUIRED")
    if get("IBKR_ENVIRONMENT") != "LIVE" or port not in LIVE_PORTS:
        errors.append("LIVE_ENVIRONMENT_OR_PORT_MISMATCH")
    if (
        min(writer_id, recon_id, quote_id) <= 0
        or len({writer_id, recon_id, quote_id}) != 3
    ):
        errors.append("LIVE_CLIENT_ID_CONFIGURATION_BLOCKED")
    if not writer_enabled:
        errors.append("LIVE_WRITE_AUTHORITY_NOT_EXPLICIT")
    if _bool(get("IBKR_LIVE_AUTOSCALE_ENABLED"), True):
        errors.append("AUTOSCALING_MUST_BE_DISABLED")
    if not (
        Decimal("0") < max_order <= policy.hard_notional_cap_eur
        and Decimal("0") < max_total <= policy.hard_notional_cap_eur
        and Decimal("0") < max_risk <= policy.maximum_risk_eur
        and max_positions == 1
        and max_daily == 1
    ):
        errors.append("LIVE_LEVEL_ONE_CAPS_BLOCKED")
    if fractional_shares_enabled:
        errors.append("FRACTIONAL_SHARES_MUST_BE_DISABLED")
    if not forbidden_products_disabled:
        errors.append("FORBIDDEN_PRODUCT_OR_LEVERAGE_FLAG")
    fingerprint_key = get("IBKR_ACCOUNT_FINGERPRINT_KEY")
    approved_fingerprint = get("IBKR_LIVE_ACCOUNT_FINGERPRINT")
    activation_phrase = get("IBKR_MANUAL_APPROVAL_PHRASE")
    if not fingerprint_key:
        errors.append("ACCOUNT_FINGERPRINT_KEY_MISSING")
    if not approved_fingerprint:
        errors.append("LIVE_ACCOUNT_FINGERPRINT_REQUIRED")
    if not activation_phrase:
        errors.append("EXACT_OPERATOR_APPROVAL_REQUIRED")
    if ttl <= 0 or callback_timeout <= 0:
        errors.append("LIVE_TIMEOUT_CONFIG_BLOCKED")
    config = LiveCanaryConfig(
        host=host,
        port=port,
        writer_client_id=writer_id,
        recon_client_id=recon_id,
        quote_client_id=quote_id,
        account_fingerprint_key=fingerprint_key,
        approved_account_fingerprint=approved_fingerprint,
        manual_activation_phrase=activation_phrase,
        writer_enabled=writer_enabled,
        max_order_eur=max_order,
        max_total_exposure_eur=max_total,
        max_risk_eur=max_risk,
        max_open_positions=max_positions,
        max_new_orders_per_day=max_daily,
        approval_ttl_seconds=ttl,
        callback_timeout_seconds=callback_timeout,
        fractional_shares_enabled=fractional_shares_enabled,
        maximum_quantity=Decimal("100"),
        canary_risk_fraction=policy.canary_risk_pct,
        maximum_stock_weight=policy.maximum_stock_weight,
        maximum_pooled_vehicle_weight=(
            policy.maximum_pooled_vehicle_weight
        ),
        maximum_portfolio_heat_fraction=(
            policy.maximum_portfolio_heat_pct
        ),
        minimum_economic_notional_eur=(
            policy.minimum_economic_notional_eur
        ),
        maximum_cost_to_expected_edge_ratio=(
            policy.maximum_cost_to_expected_edge_ratio
        ),
        policy_version=policy.policy_version,
    )
    return config, sorted(set(errors))


def load_live_portfolio_config(
    project_root: Path,
    env_file: str | Path = ".env.ibkr.portfolio.live",
) -> tuple[LiveCanaryConfig | None, list[str]]:
    """Load a separately configured, post-canary portfolio writer.

    The dedicated environment can never be inferred from the Level-1 canary
    file. Its configured euro limits must fit inside the currently resolved
    capital-level limits.
    """
    path = Path(env_file)
    if not path.is_absolute():
        path = project_root / path
    if path.name != ".env.ibkr.portfolio.live" or not path.exists():
        return None, ["DEDICATED_PORTFOLIO_LIVE_ENV_REQUIRED"]
    values = {
        key: str(value)
        for key, value in dotenv_values(path).items()
        if value is not None
    }

    def get(name: str, default: str = "") -> str:
        return values.get(name, default).strip()

    errors: list[str] = []
    host = get("IBKR_HOST")
    port = _int(get("IBKR_PORT"), -1)
    writer_id = _int(get("IBKR_CLIENT_ID"), -1)
    recon_id = _int(get("IBKR_RECON_CLIENT_ID"), -1)
    quote_id = _int(get("IBKR_QUOTE_CLIENT_ID"), -1)
    writer_enabled = (
        get("IBKR_ENVIRONMENT") == "LIVE"
        and not _bool(get("IBKR_READ_ONLY"), True)
        and get("IBKR_ORDER_AUTHORITY") == "PORTFOLIO"
        and _bool(get("IBKR_ALLOW_ORDER_TRANSMISSION"), False)
        and _bool(get("IBKR_LIVE_TRADING_ENABLED"), False)
    )
    max_order = _decimal(get("IBKR_MAX_ORDER_EUR"), Decimal("999999"))
    max_total = _decimal(
        get("IBKR_MAX_TOTAL_EXPOSURE_EUR"), Decimal("999999")
    )
    max_risk = _decimal(get("IBKR_MAX_RISK_EUR"), Decimal("999999"))
    max_positions = _int(get("IBKR_MAX_OPEN_POSITIONS"), 999)
    max_daily = _int(get("IBKR_MAX_NEW_ORDERS_PER_DAY"), 999)
    ttl = _int(get("IBKR_LIVE_APPROVAL_TTL_SECONDS"), 300)
    callback_timeout = _float(
        get("IBKR_LIVE_CALLBACK_TIMEOUT_SECONDS"), 15.0
    )
    fractional = _bool(get("IBKR_ALLOW_FRACTIONAL_SHARES"), False)
    if host not in {"127.0.0.1", "localhost"}:
        errors.append("LIVE_HOST_LOCAL_ONLY_REQUIRED")
    if get("IBKR_ENVIRONMENT") != "LIVE" or port not in LIVE_PORTS:
        errors.append("LIVE_ENVIRONMENT_OR_PORT_MISMATCH")
    if (
        min(writer_id, recon_id, quote_id) <= 0
        or len({writer_id, recon_id, quote_id}) != 3
    ):
        errors.append("LIVE_CLIENT_ID_CONFIGURATION_BLOCKED")
    if not writer_enabled:
        errors.append("LIVE_PORTFOLIO_WRITE_AUTHORITY_NOT_EXPLICIT")
    if _bool(get("IBKR_LIVE_AUTOSCALE_ENABLED"), True):
        errors.append("AUTOSCALING_MUST_BE_DISABLED")
    if fractional:
        errors.append("FRACTIONAL_SHARES_MUST_BE_DISABLED")
    if any(
        _bool(get(name), True)
        for name in (
            "IBKR_ALLOW_FUTURES",
            "IBKR_ALLOW_SHORTS",
            "IBKR_ALLOW_MARGIN",
            "IBKR_ALLOW_OPTIONS",
            "IBKR_ALLOW_FOREX_SPECULATION",
        )
    ):
        errors.append("FORBIDDEN_PRODUCT_OR_LEVERAGE_FLAG")
    if not get("IBKR_ACCOUNT_FINGERPRINT_KEY"):
        errors.append("ACCOUNT_FINGERPRINT_KEY_MISSING")
    if not get("IBKR_LIVE_ACCOUNT_FINGERPRINT"):
        errors.append("LIVE_ACCOUNT_FINGERPRINT_REQUIRED")
    if not get("IBKR_MANUAL_APPROVAL_PHRASE"):
        errors.append("EXACT_OPERATOR_APPROVAL_REQUIRED")
    if ttl <= 0 or callback_timeout <= 0:
        errors.append("LIVE_TIMEOUT_CONFIG_BLOCKED")

    capital = _read_json(project_root / "output/capital/current_level.json")
    level = int(capital.get("CURRENT_CAPITAL_LEVEL", 0) or 0)
    account = _read_json(
        project_root / "data/portfolio/private/current-state.json"
    ).get("account_state", {})
    equity = _decimal(
        str(account.get("net_liquidation_eur") or "0"), Decimal("0")
    )
    limits: dict[str, object] = {}
    if level < 2:
        errors.append("CAPITAL_LEVEL_2_REQUIRED")
    elif equity <= 0:
        errors.append("POSITIVE_ACCOUNT_EQUITY_REQUIRED")
    else:
        try:
            limits = capital_level_limits(
                project_root,
                level=level,
                account_equity_eur=equity,
            )
        except (KeyError, TypeError, ValueError):
            errors.append("CAPITAL_LEVEL_LIMITS_INVALID")
    if limits:
        permitted_order = Decimal(str(limits["maximum_stock_order_eur"]))
        permitted_total = Decimal(str(limits["maximum_total_exposure_eur"]))
        permitted_risk = Decimal(str(limits["maximum_risk_per_trade_eur"]))
        permitted_positions = int(limits["maximum_positions"])
        if not (
            Decimal("0") < max_order <= permitted_order
            and Decimal("0") < max_total <= permitted_total
            and Decimal("0") < max_risk <= permitted_risk
            and 1 <= max_positions <= permitted_positions
            and 1 <= max_daily <= 2
        ):
            errors.append("LIVE_LEVEL_TWO_CAPS_BLOCKED")
    config = LiveCanaryConfig(
        host=host,
        port=port,
        writer_client_id=writer_id,
        recon_client_id=recon_id,
        quote_client_id=quote_id,
        account_fingerprint_key=get("IBKR_ACCOUNT_FINGERPRINT_KEY"),
        approved_account_fingerprint=get("IBKR_LIVE_ACCOUNT_FINGERPRINT"),
        manual_activation_phrase=get("IBKR_MANUAL_APPROVAL_PHRASE"),
        writer_enabled=writer_enabled,
        max_order_eur=max_order,
        max_total_exposure_eur=max_total,
        max_risk_eur=max_risk,
        max_open_positions=max_positions,
        max_new_orders_per_day=max_daily,
        approval_ttl_seconds=ttl,
        callback_timeout_seconds=callback_timeout,
        fractional_shares_enabled=fractional,
        execution_authority=LIVE_LEVEL_TWO,
        maximum_quantity=Decimal("100"),
    )
    return config, sorted(set(errors))


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _bool(raw: str, default: bool) -> bool:
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _int(raw: str, default: int) -> int:
    try:
        return int(raw)
    except ValueError:
        return default


def _float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except ValueError:
        return default


def _decimal(raw: str, default: Decimal) -> Decimal:
    try:
        return Decimal(raw)
    except Exception:
        return default
