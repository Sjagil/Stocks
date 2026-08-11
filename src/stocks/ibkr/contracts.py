from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any

from stocks.application.phase_gates import PhaseGateStatus
from stocks.domain.currencies import validate_currency_code
from stocks.domain.assets import AssetClass, IbkrSecurityType, ibkr_security_type_for
from stocks.ibkr.exchanges import validate_valid_exchanges
from stocks.ibkr.futures import (
    validate_future_contract_month,
    validate_future_expiration_date,
    validate_future_last_trade_time,
)
from stocks.ibkr.market_rules import validate_market_rule_ids
from stocks.ibkr.trading_hours import validate_ibkr_hours_field
from stocks.ibkr.timezones import validate_ibkr_timezone_id


_IBKR_CODE_PATTERN = re.compile(r"^[A-Z0-9._-]+$")


@dataclass(frozen=True)
class ContractQuery:
    symbol: str
    asset_class: AssetClass
    currency: str
    exchange: str = "SMART"
    primary_exchange: str | None = None
    expiry: str | None = None

    @property
    def security_type(self) -> IbkrSecurityType:
        return ibkr_security_type_for(self.asset_class)

    def validate(self) -> None:
        _validate_canonical_ibkr_code(self.symbol, "symbol")
        _validate_canonical_ibkr_code(self.exchange, "exchange")
        if self.primary_exchange is not None:
            _validate_canonical_ibkr_code(self.primary_exchange, "primary_exchange")
        _reject_surrounding_whitespace(self.currency, "currency")
        validate_currency_code(self.currency)
        if self.security_type == IbkrSecurityType.FUT and not self.expiry:
            raise ValueError("futures require explicit expiry before qualification")
        if self.expiry is not None:
            _reject_surrounding_whitespace(self.expiry, "expiry")
        if self.asset_class == AssetClass.CASH:
            raise ValueError("cash is allocator state, not an IBKR contract query")


@dataclass(frozen=True)
class ResolvedContract:
    con_id: int
    symbol: str
    local_symbol: str
    security_type: IbkrSecurityType
    exchange: str
    currency: str
    trading_class: str | None
    primary_exchange: str | None = None
    multiplier: Decimal | None = None
    min_tick: Decimal | None = None
    expiry: str | None = None
    valid_exchanges: str | None = None
    market_rule_ids: str | None = None
    long_name: str | None = None
    industry: str | None = None
    category: str | None = None
    subcategory: str | None = None
    last_trade_date_or_contract_month: str | None = None
    real_expiration_date: str | None = None
    last_trade_time: str | None = None
    under_con_id: int | None = None
    first_notice_day: date | None = None
    last_trade_day: date | None = None
    delivery_type: str | None = None
    settlement_type: str | None = None
    contract_size_unit: str | None = None
    roll_group: str | None = None
    time_zone_id: str | None = None
    liquid_hours: str | None = None
    trading_hours: str | None = None

    def validate_for_storage(self) -> None:
        if self.con_id <= 0:
            raise ValueError("con_id must be positive")
        if not self.local_symbol:
            raise ValueError("local_symbol is required")
        validate_currency_code(self.currency)
        if self.security_type == IbkrSecurityType.FUT:
            if not self.expiry:
                raise ValueError("resolved futures require expiry")
            if self.multiplier is None or self.multiplier <= 0:
                raise ValueError("resolved futures require positive multiplier")

    def validate_phase2_required_fields(self) -> None:
        self.validate_for_storage()
        common_missing = _missing_required_text_fields(
            {
                "symbol": self.symbol,
                "local_symbol": self.local_symbol,
                "exchange": self.exchange,
                "currency": self.currency,
                "trading_class": self.trading_class,
                "min_tick": self.min_tick,
                "time_zone_id": self.time_zone_id,
                "trading_hours": self.trading_hours,
                "liquid_hours": self.liquid_hours,
                "market_rule_ids": self.market_rule_ids,
            }
        )
        if common_missing:
            raise ValueError(f"missing required contract fields: {', '.join(common_missing)}")
        if self.min_tick is None or self.min_tick <= 0:
            raise ValueError("min_tick must be positive")

        validate_ibkr_timezone_id(self.time_zone_id)
        validate_ibkr_hours_field("trading_hours", self.trading_hours)
        validate_ibkr_hours_field("liquid_hours", self.liquid_hours)
        validate_market_rule_ids(self.market_rule_ids)

        if self.security_type == IbkrSecurityType.STK:
            stk_missing = _missing_required_text_fields(
                {
                    "primary_exchange": self.primary_exchange,
                    "valid_exchanges": self.valid_exchanges,
                    "long_name": self.long_name,
                }
            )
            if stk_missing:
                raise ValueError(f"missing required STK fields: {', '.join(stk_missing)}")
            validate_valid_exchanges(self.valid_exchanges, primary_exchange=self.primary_exchange)

        if self.security_type == IbkrSecurityType.FUT:
            fut_missing = _missing_required_text_fields(
                {
                    "last_trade_date_or_contract_month": self.last_trade_date_or_contract_month,
                    "real_expiration_date": self.real_expiration_date,
                    "last_trade_time": self.last_trade_time,
                    "under_con_id": self.under_con_id,
                }
            )
            if fut_missing:
                raise ValueError(f"missing required FUT fields: {', '.join(fut_missing)}")
            if self.under_con_id is None or self.under_con_id <= 0:
                raise ValueError("under_con_id must be positive")
            validate_future_contract_month(self.expiry, field_name="expiry")
            validate_future_contract_month(
                self.last_trade_date_or_contract_month,
                field_name="last_trade_date_or_contract_month",
            )
            validate_future_expiration_date(self.real_expiration_date)
            validate_future_last_trade_time(self.last_trade_time)
            _validate_future_reference_metadata(self)
        else:
            _reject_future_reference_metadata(self)


class ContractResolverNotEnabled(RuntimeError):
    """Raised when live Phase 2 contract requests are attempted before Phase 1 freeze."""


class OfflineContractResolver:
    def qualify(self, query: ContractQuery) -> ResolvedContract:
        query.validate()
        raise ContractResolverNotEnabled(
            "Phase 2 live contract resolution is disabled until Phase 1 is frozen."
        )


class ContractResolutionStatus(str, Enum):
    PHASE1_NOT_FROZEN = "PHASE1_NOT_FROZEN"
    LIVE_RESOLVER_NOT_IMPLEMENTED = "LIVE_RESOLVER_NOT_IMPLEMENTED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_CURRENCY = "INVALID_CURRENCY"
    INVALID_EXCHANGE = "INVALID_EXCHANGE"
    MISSING_REQUIRED_FIELDS = "MISSING_REQUIRED_FIELDS"
    STALE_CACHE = "STALE_CACHE"
    CALLBACK_TIMEOUT = "CALLBACK_TIMEOUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    NOT_FOUND = "NOT_FOUND"
    RESOLVED = "RESOLVED"
    AMBIGUOUS_BLOCKED = "AMBIGUOUS_BLOCKED"
    CONTRACT_VALIDATION_FAILED = "CONTRACT_VALIDATION_FAILED"


@dataclass(frozen=True)
class StockContractRequest:
    symbol: str
    currency: str
    exchange: str
    primary_exchange: str
    security_type: IbkrSecurityType = IbkrSecurityType.STK
    asset_class: AssetClass = AssetClass.STOCK

    def to_resolution_request(self) -> ContractResolutionRequest:
        return ContractResolutionRequest(
            symbol=self.symbol,
            asset_class=self.asset_class,
            security_type=self.security_type,
            currency=self.currency,
            exchange=self.exchange,
            primary_exchange=self.primary_exchange,
        )


@dataclass(frozen=True)
class FutureContractRequest:
    symbol: str
    currency: str
    exchange: str
    expiry: str | None = None
    security_type: IbkrSecurityType = IbkrSecurityType.FUT
    asset_class: AssetClass = AssetClass.COMMODITY_FUTURE

    def to_resolution_request(self) -> ContractResolutionRequest:
        return ContractResolutionRequest(
            symbol=self.symbol,
            asset_class=self.asset_class,
            security_type=self.security_type,
            currency=self.currency,
            exchange=self.exchange,
            expiry=self.expiry,
        )


@dataclass(frozen=True)
class ContractResolutionRequest:
    symbol: str
    asset_class: AssetClass
    security_type: IbkrSecurityType
    currency: str
    exchange: str
    primary_exchange: str | None = None
    expiry: str | None = None

    def validate_basic(self) -> None:
        _validate_canonical_ibkr_code(self.symbol, "symbol")
        _validate_canonical_ibkr_code(self.exchange, "exchange")
        if self.primary_exchange is not None:
            _validate_canonical_ibkr_code(self.primary_exchange, "primary_exchange")
        _reject_surrounding_whitespace(self.currency, "currency")
        validate_currency_code(self.currency)
        if self.asset_class == AssetClass.CASH:
            raise ValueError("cash is allocator state, not an IBKR contract query")
        expected_security_type = ibkr_security_type_for(self.asset_class)
        if self.security_type != expected_security_type:
            raise ValueError(
                f"asset_class {self.asset_class.value} requires secType {expected_security_type.value}, "
                f"got {self.security_type.value}"
        )
        if self.security_type == IbkrSecurityType.FUT and self.expiry:
            _reject_surrounding_whitespace(self.expiry, "expiry")
            validate_future_contract_month(self.expiry, field_name="expiry")
        elif self.expiry:
            raise ValueError("expiry is only valid for futures contract requests")

    def as_dict(self) -> dict[str, str | None]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class.value,
            "security_type": self.security_type.value,
            "currency": self.currency,
            "exchange": self.exchange,
            "primary_exchange": self.primary_exchange,
            "expiry": self.expiry,
        }


def canonical_request_payload(request: ContractResolutionRequest) -> dict[str, Any]:
    request.validate_basic()
    return request.as_dict()


def contract_request_hash(request: ContractResolutionRequest) -> str:
    payload = canonical_request_payload(request)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()


class ContractRequestIdAllocator:
    def __init__(self, *, start: int = 1_000_000) -> None:
        if start <= 0:
            raise ValueError("request id allocator start must be positive")
        self._next_id = start
        self._lock = threading.Lock()

    def next_id(self) -> int:
        with self._lock:
            value = self._next_id
            self._next_id += 1
            return value


@dataclass(frozen=True)
class IbkrContractSpec:
    symbol: str
    security_type: IbkrSecurityType
    exchange: str
    currency: str
    primary_exchange: str | None = None
    last_trade_date_or_contract_month: str | None = None

    def as_ibapi_fields(self) -> dict[str, str]:
        fields = {
            "symbol": self.symbol,
            "secType": self.security_type.value,
            "exchange": self.exchange,
            "currency": self.currency,
        }
        if self.primary_exchange:
            fields["primaryExchange"] = self.primary_exchange
        if self.last_trade_date_or_contract_month:
            fields["lastTradeDateOrContractMonth"] = self.last_trade_date_or_contract_month
        return fields

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "ibkr_contract_spec_v1",
            "ibapi_fields": self.as_ibapi_fields(),
            "financial_calls": {
                "place_order": 0,
                "cancel_order": 0,
                "global_cancel": 0,
            },
        }


@dataclass(frozen=True)
class ContractResolutionReport:
    status: ContractResolutionStatus
    request: ContractResolutionRequest
    phase1: PhaseGateStatus
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "ibkr_contract_resolution_v1",
            "status": self.status.value,
            "phase1": self.phase1.as_dict(),
            "request": self.request.as_dict(),
            "resolved_contract": None,
            "financial_calls": {
                "place_order": 0,
                "cancel_order": 0,
                "global_cancel": 0,
            },
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ContractCandidateEvaluation:
    status: ContractResolutionStatus
    matches: tuple[ResolvedContract, ...]
    rejected_count: int
    reason: str
    rejected_candidates: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "returned_match_count": len(self.matches),
            "rejected_count": self.rejected_count,
            "selected_conId": self.matches[0].con_id if self.status == ContractResolutionStatus.RESOLVED else None,
            "reason": self.reason,
            "matched_candidates": [_candidate_summary(candidate) for candidate in self.matches],
            "rejected_candidates": list(self.rejected_candidates),
        }


def gated_contract_resolution_report(
    request: ContractResolutionRequest,
    phase1: PhaseGateStatus,
) -> ContractResolutionReport:
    try:
        request.validate_basic()
    except ValueError as exc:
        return ContractResolutionReport(
            status=ContractResolutionStatus.VALIDATION_ERROR,
            request=request,
            phase1=phase1,
            reason=str(exc),
        )

    if not phase1.frozen:
        return ContractResolutionReport(
            status=ContractResolutionStatus.PHASE1_NOT_FROZEN,
            request=request,
            phase1=phase1,
            reason="Phase 2 contract resolution is blocked until Phase 1 freeze is proven.",
        )

    return ContractResolutionReport(
        status=ContractResolutionStatus.LIVE_RESOLVER_NOT_IMPLEMENTED,
        request=request,
        phase1=phase1,
        reason="Live contract resolution is the next Phase 2 implementation step.",
    )


def build_ibkr_contract_spec(request: ContractResolutionRequest) -> IbkrContractSpec:
    request.validate_basic()
    if request.security_type == IbkrSecurityType.STK:
        return IbkrContractSpec(
            symbol=request.symbol.upper(),
            security_type=IbkrSecurityType.STK,
            exchange=request.exchange.upper(),
            currency=request.currency.upper(),
            primary_exchange=request.primary_exchange.upper() if request.primary_exchange else None,
        )

    if request.security_type == IbkrSecurityType.FUT:
        if not request.expiry:
            return IbkrContractSpec(
                symbol=request.symbol.upper(),
                security_type=IbkrSecurityType.FUT,
                exchange=request.exchange.upper(),
                currency=request.currency.upper(),
            )
        if not _valid_ibkr_contract_month(request.expiry):
            raise ValueError("futures expiry must use IBKR YYYYMM or YYYYMMDD format")
        return IbkrContractSpec(
            symbol=request.symbol.upper(),
            security_type=IbkrSecurityType.FUT,
            exchange=request.exchange.upper(),
            currency=request.currency.upper(),
            last_trade_date_or_contract_month=request.expiry,
        )

    raise ValueError(f"unsupported security type for Phase 2 contract spec: {request.security_type.value}")


def build_native_ibapi_contract(spec: IbkrContractSpec) -> Any:
    from ibapi.contract import Contract

    contract = Contract()
    for field_name, value in spec.as_ibapi_fields().items():
        setattr(contract, field_name, value)
    return contract


def resolved_contract_from_contract_details(contract_details: Any) -> ResolvedContract:
    native_contract = contract_details.contract
    security_type = IbkrSecurityType(_required_text(native_contract.secType, "secType"))
    last_trade_date_or_contract_month = _optional_text(
        getattr(native_contract, "lastTradeDateOrContractMonth", None)
    ) or _optional_text(getattr(contract_details, "contractMonth", None))
    expiry = last_trade_date_or_contract_month if security_type == IbkrSecurityType.FUT else None

    return ResolvedContract(
        con_id=int(native_contract.conId),
        symbol=_required_text(native_contract.symbol, "symbol"),
        local_symbol=_required_text(native_contract.localSymbol, "localSymbol"),
        security_type=security_type,
        exchange=_required_text(native_contract.exchange, "exchange"),
        primary_exchange=_optional_text(getattr(native_contract, "primaryExchange", None)),
        currency=_required_text(native_contract.currency, "currency"),
        trading_class=_optional_text(getattr(native_contract, "tradingClass", None)),
        multiplier=_optional_decimal(getattr(native_contract, "multiplier", None)),
        min_tick=_optional_decimal(getattr(contract_details, "minTick", None)),
        expiry=expiry,
        valid_exchanges=_optional_text(getattr(contract_details, "validExchanges", None)),
        market_rule_ids=_optional_text(getattr(contract_details, "marketRuleIds", None)),
        long_name=_optional_text(getattr(contract_details, "longName", None)),
        industry=_optional_text(getattr(contract_details, "industry", None)),
        category=_optional_text(getattr(contract_details, "category", None)),
        subcategory=_optional_text(getattr(contract_details, "subcategory", None)),
        last_trade_date_or_contract_month=last_trade_date_or_contract_month,
        real_expiration_date=_optional_text(getattr(contract_details, "realExpirationDate", None)),
        last_trade_time=_optional_text(getattr(contract_details, "lastTradeTime", None)),
        under_con_id=_optional_int(getattr(contract_details, "underConId", None)),
        time_zone_id=_optional_text(getattr(contract_details, "timeZoneId", None)),
        trading_hours=_optional_text(getattr(contract_details, "tradingHours", None)),
        liquid_hours=_optional_text(getattr(contract_details, "liquidHours", None)),
    )


def classify_contract_match_count(match_count: int) -> ContractResolutionStatus:
    if match_count < 0:
        raise ValueError("match_count cannot be negative")
    if match_count == 0:
        return ContractResolutionStatus.NOT_FOUND
    if match_count == 1:
        return ContractResolutionStatus.RESOLVED
    return ContractResolutionStatus.AMBIGUOUS_BLOCKED


def evaluate_contract_candidates(
    request: ContractResolutionRequest,
    candidates: list[ResolvedContract],
) -> ContractCandidateEvaluation:
    request.validate_basic()
    valid_matches: list[ResolvedContract] = []
    rejected_candidates: list[dict[str, Any]] = []

    for candidate in candidates:
        try:
            candidate.validate_for_storage()
        except ValueError as exc:
            rejected_candidates.append(
                _candidate_rejection(candidate, [f"storage validation failed: {exc}"])
            )
            continue

        rejection_reasons = _candidate_rejection_reasons(candidate, request)
        if rejection_reasons:
            rejected_candidates.append(_candidate_rejection(candidate, rejection_reasons))
            continue
        valid_matches.append(candidate)

    status = classify_contract_match_count(len(valid_matches))
    if status == ContractResolutionStatus.NOT_FOUND:
        reason = "no candidate matched symbol, security type, currency, exchange and expiry filters"
    elif status == ContractResolutionStatus.RESOLVED:
        reason = "exactly one valid contract candidate matched"
    else:
        reason = "multiple valid contract candidates matched; manual disambiguation required"

    return ContractCandidateEvaluation(
        status=status,
        matches=tuple(valid_matches),
        rejected_count=len(rejected_candidates),
        reason=reason,
        rejected_candidates=tuple(rejected_candidates),
    )


def evaluate_contract_details_payloads(
    request: ContractResolutionRequest,
    contract_details_items: list[Any],
) -> ContractCandidateEvaluation:
    candidates: list[ResolvedContract] = []
    rejected_count = 0
    for contract_details in contract_details_items:
        try:
            candidates.append(resolved_contract_from_contract_details(contract_details))
        except (AttributeError, TypeError, ValueError):
            rejected_count += 1

    evaluation = evaluate_contract_candidates(request, candidates)
    return ContractCandidateEvaluation(
        status=evaluation.status,
        matches=evaluation.matches,
        rejected_count=evaluation.rejected_count + rejected_count,
        reason=evaluation.reason,
        rejected_candidates=evaluation.rejected_candidates,
    )


def _candidate_matches_request(candidate: ResolvedContract, request: ContractResolutionRequest) -> bool:
    return not _candidate_rejection_reasons(candidate, request)


def _candidate_rejection_reasons(candidate: ResolvedContract, request: ContractResolutionRequest) -> list[str]:
    reasons: list[str] = []
    if candidate.symbol.upper() != request.symbol.upper():
        reasons.append("symbol")
    if candidate.security_type != request.security_type:
        reasons.append("security_type")
    if candidate.currency.upper() != request.currency.upper():
        reasons.append("currency")
    if candidate.exchange.upper() != request.exchange.upper() and request.exchange.upper() != "SMART":
        reasons.append("exchange")
    if request.primary_exchange and (candidate.primary_exchange or "").upper() != request.primary_exchange.upper():
        reasons.append("primary_exchange")
    if request.expiry and candidate.expiry != request.expiry:
        reasons.append("expiry")
    return reasons


def _candidate_rejection(candidate: ResolvedContract, reasons: list[str]) -> dict[str, Any]:
    return {
        **_candidate_summary(candidate),
        "reasons": reasons,
    }


def _candidate_summary(candidate: ResolvedContract) -> dict[str, Any]:
    return {
        "con_id": candidate.con_id,
        "symbol": candidate.symbol,
        "local_symbol": candidate.local_symbol,
        "security_type": candidate.security_type.value,
        "currency": candidate.currency,
        "exchange": candidate.exchange,
        "primary_exchange": candidate.primary_exchange,
        "trading_class": candidate.trading_class,
        "expiry": candidate.expiry,
    }


def _validate_canonical_ibkr_code(value: str, field_name: str) -> None:
    _reject_surrounding_whitespace(value, field_name)
    if not value:
        raise ValueError(f"{field_name} is required")
    if not _IBKR_CODE_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be an uppercase IBKR code")


def _reject_surrounding_whitespace(value: str, field_name: str) -> None:
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")


def _missing_required_text_fields(fields: dict[str, object]) -> list[str]:
    missing: list[str] = []
    for name, value in fields.items():
        if value is None:
            missing.append(name)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(name)
    return missing


def _validate_future_reference_metadata(contract: ResolvedContract) -> None:
    missing = _missing_required_text_fields(
        {
            "delivery_type": contract.delivery_type,
            "settlement_type": contract.settlement_type,
            "contract_size_unit": contract.contract_size_unit,
            "roll_group": contract.roll_group,
        }
    )
    supplied = [
        field_name
        for field_name in (
            "delivery_type",
            "settlement_type",
            "contract_size_unit",
            "roll_group",
        )
        if getattr(contract, field_name) is not None
    ]
    if supplied and missing:
        raise ValueError(f"incomplete FUT reference metadata: {', '.join(missing)}")


def _reject_future_reference_metadata(contract: ResolvedContract) -> None:
    supplied = [
        field_name
        for field_name in (
            "delivery_type",
            "settlement_type",
            "contract_size_unit",
            "roll_group",
        )
        if getattr(contract, field_name) is not None
    ]
    if supplied:
        raise ValueError(f"FUT reference metadata is only valid for futures: {', '.join(supplied)}")


def _valid_ibkr_contract_month(value: str) -> bool:
    return bool(re.fullmatch(r"\d{6}(\d{2})?", value))


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_decimal(value: object) -> Decimal | None:
    text = _optional_text(value)
    if text is None:
        return None
    return Decimal(text).normalize()


def _optional_int(value: object) -> int | None:
    text = _optional_text(value)
    if text is None:
        return None
    return int(text)
