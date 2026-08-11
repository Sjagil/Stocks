from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


ACCOUNT_LIFECYCLE_STATES = (
    "DISCONNECTED",
    "CONNECTED",
    "ACCOUNT_SYNCING",
    "ACCOUNT_PARTIAL",
    "ACCOUNT_READY",
    "ACCOUNT_STALE",
    "ACCOUNT_INVALID",
)
ECONOMIC_ACCOUNT_TAGS = (
    "NetLiquidation",
    "TotalCashValue",
    "SettledCash",
    "AvailableFunds",
    "BuyingPower",
    "GrossPositionValue",
    "InitMarginReq",
    "MaintMarginReq",
    "ExcessLiquidity",
)
RESEARCH_REQUIRED_TAGS = (
    "NetLiquidation",
    "TotalCashValue",
    "AvailableFunds",
)
EXECUTION_REQUIRED_TAGS = tuple(
    tag for tag in ECONOMIC_ACCOUNT_TAGS if tag != "SettledCash"
)
ACTIVE_ORDER_STATUSES = {
    "PENDINGSUBMIT",
    "APIPENDING",
    "PRESUBMITTED",
    "SUBMITTED",
    "PENDINGCANCEL",
}


def derive_economic_account_state(
    snapshot: Mapping[str, Any] | None,
    *,
    expected_base_currency: str,
    snapshot_hash_verified: bool,
    now: datetime | None = None,
    execution_max_age: timedelta = timedelta(minutes=2),
    fx_to_eur: Mapping[str, Decimal] | None = None,
) -> dict[str, Any]:
    """Derive one financial read model from a canonical broker snapshot.

    Aggregate account-summary EUR values are reporting values. Only direct
    ledger cash in EUR can become SPENDABLE_EUR; no implicit FX conversion is
    assumed.
    """
    current = (now or datetime.now(UTC)).astimezone(UTC)
    if not snapshot:
        return _blocked("DISCONNECTED", ["BROKER_SNAPSHOT_UNAVAILABLE"])
    values = list(snapshot.get("account", {}).get("values", []))
    account_status = str(snapshot.get("account", {}).get("status", ""))
    position_status = str(snapshot.get("positions", {}).get("status", ""))
    same_order_status = str(
        snapshot.get("same_client_open_orders", {}).get("status", "")
    )
    all_order_status = str(
        snapshot.get("all_api_open_orders", {}).get("status", "")
    )
    execution_status = str(snapshot.get("executions", {}).get("status", ""))
    fingerprints = {
        str(row.get("account_fingerprint"))
        for row in values
        if row.get("account_fingerprint")
        and not str(row.get("tag", "")).startswith("$LEDGER-")
    }
    parsed_values, invalid_value_tags = _account_values(values)
    expected = expected_base_currency.strip().upper()
    aggregate_currencies = {
        currency
        for tag, currency in parsed_values
        if tag in ECONOMIC_ACCOUNT_TAGS and currency not in {"", "BASE"}
    }
    base_currency_valid = bool(expected) and (
        not aggregate_currencies or expected in aggregate_currencies
    )
    base_currency = expected if base_currency_valid else "UNKNOWN"
    aggregate = {
        tag: _preferred_value(parsed_values, tag, expected)
        for tag in ECONOMIC_ACCOUNT_TAGS
    }
    cash_by_currency = _currency_values(
        parsed_values,
        tags=("CashBalance", "TotalCashBalance"),
    )
    settled_cash_by_currency = _currency_values(
        parsed_values,
        tags=("SettledCash",),
        direct_only=bool(cash_by_currency),
    )
    reporting_value_eur = (
        aggregate["NetLiquidation"] if expected == "EUR" else None
    )
    reporting_cash_eur = (
        aggregate["TotalCashValue"] if expected == "EUR" else None
    )
    spendable_eur = cash_by_currency.get("EUR")
    reserved = _open_order_reserved_capital_eur(
        snapshot,
        fx_to_eur=fx_to_eur or {"EUR": Decimal("1")},
    )
    buying_capacity = aggregate["AvailableFunds"]
    excess = aggregate["ExcessLiquidity"]
    execution_capacity: Decimal | None = None
    if (
        spendable_eur is not None
        and buying_capacity is not None
        and excess is not None
        and reserved["complete"]
    ):
        execution_capacity = max(
            Decimal("0"),
            min(spendable_eur, buying_capacity, excess)
            - reserved["reserved_eur"],
        )
    research_capacity = _minimum_known(
        aggregate["AvailableFunds"], aggregate["TotalCashValue"]
    )
    account_updated_at = _latest_value_timestamp(values)
    portfolio_updated_at = _component_timestamp(snapshot, "positions")
    order_updated_at = _latest_component_timestamp(
        snapshot,
        ("same_client_open_orders", "all_api_open_orders"),
    )
    position_reconciled_at = portfolio_updated_at
    account_fresh = _fresh(account_updated_at, current, execution_max_age)
    portfolio_fresh = _fresh(
        portfolio_updated_at, current, execution_max_age
    )
    orders_fresh = _fresh(order_updated_at, current, execution_max_age)
    account_complete = account_status == "COMPLETE"
    position_complete = position_status in {"COMPLETE", "EMPTY_COMPLETE"}
    orders_complete = same_order_status in {
        "COMPLETE",
        "EMPTY_COMPLETE",
    } and all_order_status in {"COMPLETE", "EMPTY_COMPLETE"}
    executions_complete = execution_status in {"COMPLETE", "EMPTY_COMPLETE"}
    missing_research = [
        tag for tag in RESEARCH_REQUIRED_TAGS if aggregate[tag] is None
    ]
    missing_execution = [
        tag for tag in EXECUTION_REQUIRED_TAGS if aggregate[tag] is None
    ]
    research_ready = (
        account_complete
        and len(fingerprints) == 1
        and not missing_research
        and not invalid_value_tags
        and reporting_value_eur is not None
        and research_capacity is not None
    )
    execution_blockers: list[str] = []
    if not snapshot_hash_verified:
        execution_blockers.append("ACCOUNT_SNAPSHOT_HASH_UNVERIFIED")
    if not account_complete:
        execution_blockers.append("ACCOUNT_DOWNLOAD_INCOMPLETE")
    if not position_complete:
        execution_blockers.append("POSITION_DOWNLOAD_INCOMPLETE")
    if not orders_complete:
        execution_blockers.append("ORDER_DOWNLOAD_INCOMPLETE")
    if not executions_complete:
        execution_blockers.append("EXECUTION_DOWNLOAD_INCOMPLETE")
    if len(fingerprints) != 1:
        execution_blockers.append("ACCOUNT_FINGERPRINT_NOT_UNIQUE")
    if not base_currency_valid:
        execution_blockers.append("ACCOUNT_BASE_CURRENCY_UNVERIFIED")
    if missing_execution:
        execution_blockers.append("REQUIRED_ECONOMIC_ACCOUNT_TAGS_MISSING")
    if invalid_value_tags:
        execution_blockers.append("INVALID_ECONOMIC_ACCOUNT_VALUE")
    if spendable_eur is None:
        execution_blockers.append("DIRECT_SPENDABLE_EUR_NOT_PROVEN")
    if not reserved["complete"]:
        execution_blockers.append("OPEN_ORDER_RESERVED_CAPITAL_UNRESOLVED")
    if not account_fresh:
        execution_blockers.append("ACCOUNT_STATE_STALE")
    if not portfolio_fresh:
        execution_blockers.append("POSITION_STATE_STALE")
    if not orders_fresh:
        execution_blockers.append("ORDER_STATE_STALE")
    execution_ready = not execution_blockers and execution_capacity is not None
    if not snapshot.get("server_version"):
        lifecycle = "DISCONNECTED"
    elif not account_complete and not values:
        lifecycle = "ACCOUNT_SYNCING"
    elif not account_complete or missing_research:
        lifecycle = "ACCOUNT_PARTIAL"
    elif invalid_value_tags or len(fingerprints) != 1:
        lifecycle = "ACCOUNT_INVALID"
    elif not account_fresh or not portfolio_fresh or not orders_fresh:
        lifecycle = "ACCOUNT_STALE"
    else:
        lifecycle = "ACCOUNT_READY"
    return {
        "schema": "ibkr_economic_account_state_v1",
        "lifecycle_state": lifecycle,
        "lifecycle_states": list(ACCOUNT_LIFECYCLE_STATES),
        "research_status": "RESEARCH_READY" if research_ready else "NO_GO",
        "execution_status": (
            "EXECUTION_ACCOUNT_READY" if execution_ready else "NO_GO"
        ),
        "account_base_currency": base_currency,
        "base_currency_source": (
            "CONFIG_VALIDATED_BY_SUMMARY_CURRENCY"
            if base_currency_valid
            else "UNVERIFIED"
        ),
        "cash_by_currency": _string_map(cash_by_currency),
        "settled_cash_by_currency": _string_map(settled_cash_by_currency),
        "available_funds": _money(aggregate["AvailableFunds"], expected),
        "buying_power": _money(aggregate["BuyingPower"], expected),
        "net_liquidation": _money(aggregate["NetLiquidation"], expected),
        "gross_position_value": _money(
            aggregate["GrossPositionValue"], expected
        ),
        "maint_margin": _money(aggregate["MaintMarginReq"], expected),
        "init_margin": _money(aggregate["InitMarginReq"], expected),
        "excess_liquidity": _money(
            aggregate["ExcessLiquidity"], expected
        ),
        "open_order_reserved_capital_eur": str(reserved["reserved_eur"]),
        "open_order_reserved_capital_complete": reserved["complete"],
        "reporting_value_eur": _string(reporting_value_eur),
        "reporting_cash_eur": _string(reporting_cash_eur),
        "spendable_eur": _string(spendable_eur),
        "eur_available_for_new_longs": _string(execution_capacity),
        "research_sizing_capacity_eur": _string(research_capacity),
        "execution_sizing_capacity_eur": _string(execution_capacity),
        "buying_power_is_cash": False,
        "net_liquidation_is_deployable_cash": False,
        "implicit_fx_conversion_assumed": False,
        "non_eur_cash_excluded_from_spendable_eur": True,
        "account_update_completed": account_complete,
        "portfolio_update_completed": position_complete,
        "order_reconciliation_completed": orders_complete,
        "position_reconciliation_completed": position_complete,
        "execution_reconciliation_completed": executions_complete,
        "last_account_update": _iso(account_updated_at),
        "last_portfolio_update": _iso(portfolio_updated_at),
        "last_order_reconciliation": _iso(order_updated_at),
        "last_position_reconciliation": _iso(position_reconciled_at),
        "account_state_fresh": account_fresh,
        "position_state_fresh": portfolio_fresh,
        "order_state_fresh": orders_fresh,
        "missing_research_tags": missing_research,
        "missing_execution_tags": missing_execution,
        "invalid_value_tags": sorted(invalid_value_tags),
        "execution_blockers": sorted(set(execution_blockers)),
        "account_fingerprint_count": len(fingerprints),
        "raw_account_id_stored": False,
        "automatic_fx_conversion": False,
        "execution_authority": "NONE",
    }


def _account_values(
    rows: list[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], Decimal], set[str]]:
    values: dict[tuple[str, str], Decimal] = {}
    invalid: set[str] = set()
    for row in rows:
        raw_tag = str(row.get("tag") or "")
        tag = (
            raw_tag.removeprefix("$LEDGER-")
            if raw_tag.startswith("$LEDGER-")
            else raw_tag
        )
        if tag not in set(ECONOMIC_ACCOUNT_TAGS) | {
            "CashBalance",
            "TotalCashBalance",
        }:
            continue
        try:
            value = Decimal(str(row.get("value")))
        except (InvalidOperation, ValueError):
            invalid.add(tag)
            continue
        if not value.is_finite():
            invalid.add(tag)
            continue
        key = (tag, str(row.get("currency") or "").upper())
        if key in values and values[key] != value:
            invalid.add(tag)
            continue
        values[key] = value
    return values, invalid


def _preferred_value(
    values: Mapping[tuple[str, str], Decimal],
    tag: str,
    currency: str,
) -> Decimal | None:
    for key in ((tag, currency), (tag, "BASE")):
        if key in values:
            return values[key]
    return None


def _currency_values(
    values: Mapping[tuple[str, str], Decimal],
    *,
    tags: tuple[str, ...],
    direct_only: bool = True,
) -> dict[str, Decimal]:
    output: dict[str, Decimal] = {}
    for (tag, currency), value in values.items():
        if tag not in tags or currency in {"", "BASE"}:
            continue
        if direct_only and len(currency) != 3:
            continue
        output[currency] = min(output.get(currency, value), value)
    return output


def _open_order_reserved_capital_eur(
    snapshot: Mapping[str, Any],
    *,
    fx_to_eur: Mapping[str, Decimal],
) -> dict[str, Any]:
    orders = snapshot.get("all_api_open_orders", {}).get("open_orders", [])
    reserved = Decimal("0")
    complete = True
    seen: set[str] = set()
    for order in orders:
        if str(order.get("action", "")).upper() != "BUY":
            continue
        if str(order.get("order_status", "")).upper() not in ACTIVE_ORDER_STATUSES:
            continue
        identity = str(order.get("perm_id") or order.get("broker_order_id") or "")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        remaining = _decimal(order.get("remaining_quantity"))
        price = _decimal(order.get("limit_price"))
        currency = str(order.get("currency") or "").upper()
        fx = fx_to_eur.get(currency)
        if remaining is None or remaining < 0 or price is None or price <= 0:
            complete = False
            continue
        if fx is None or fx <= 0:
            complete = False
            continue
        reserved += remaining * price * fx
    return {"reserved_eur": reserved, "complete": complete}


def _latest_value_timestamp(rows: list[Mapping[str, Any]]) -> datetime | None:
    timestamps = [_timestamp(row.get("observed_at")) for row in rows]
    valid = [value for value in timestamps if value is not None]
    return max(valid) if valid else None


def _component_timestamp(
    snapshot: Mapping[str, Any], name: str
) -> datetime | None:
    component = snapshot.get("component_timestamps", {}).get(name, {})
    return _timestamp(component.get("completed_at"))


def _latest_component_timestamp(
    snapshot: Mapping[str, Any], names: tuple[str, ...]
) -> datetime | None:
    values = [_component_timestamp(snapshot, name) for name in names]
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def _timestamp(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _fresh(
    timestamp: datetime | None,
    now: datetime,
    maximum_age: timedelta,
) -> bool:
    return bool(
        timestamp is not None
        and -timedelta(seconds=5) <= now - timestamp <= maximum_age
    )


def _minimum_known(*values: Decimal | None) -> Decimal | None:
    known = [value for value in values if value is not None]
    return min(known) if len(known) == len(values) else None


def _decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _money(value: Decimal | None, currency: str) -> dict[str, str | None]:
    return {"value": _string(value), "currency": currency or None}


def _string(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _string_map(values: Mapping[str, Decimal]) -> dict[str, str]:
    return {key: str(value) for key, value in sorted(values.items())}


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _blocked(lifecycle: str, blockers: list[str]) -> dict[str, Any]:
    return {
        "schema": "ibkr_economic_account_state_v1",
        "lifecycle_state": lifecycle,
        "lifecycle_states": list(ACCOUNT_LIFECYCLE_STATES),
        "research_status": "NO_GO",
        "execution_status": "NO_GO",
        "execution_blockers": blockers,
        "execution_authority": "NONE",
        "raw_account_id_stored": False,
    }


__all__ = [
    "ACCOUNT_LIFECYCLE_STATES",
    "ECONOMIC_ACCOUNT_TAGS",
    "derive_economic_account_state",
]
