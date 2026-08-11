from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from stocks.ibkr.connection import ReadOnlyIbkrConnectionService
from stocks.ibkr.contract_cache import (
    ContractCacheLayout,
    build_contract_resolution_artifacts,
    build_resolution_audit_record,
    contract_identity_document,
    find_fresh_contract_cache_hit,
    initialize_contract_cache,
    persist_contract_resolution_artifacts,
    read_contract_cache_rows,
    validate_contract_cache,
)
from stocks.ibkr.contracts import (
    ContractCandidateEvaluation,
    ContractRequestIdAllocator,
    ContractResolutionRequest,
    ContractResolutionStatus,
    build_ibkr_contract_spec,
    build_native_ibapi_contract,
    evaluate_contract_details_payloads,
)
from stocks.ibkr.health import HealthStatus


READ_ONLY_CONTRACT_CALLS_ZERO = {
    "req_matching_symbols": 0,
    "req_contract_details": 0,
    "req_market_rule": 0,
    "req_mkt_data": 0,
    "req_historical_data": 0,
}


@dataclass(frozen=True)
class LiveContractResolutionResult:
    status: ContractResolutionStatus
    request: ContractResolutionRequest
    request_id: int | None
    source: str
    reason: str
    returned_match_count: int
    rejected_count: int
    cache_hit: bool
    persisted: dict[str, Any] | None = None
    resolved_contract_identity: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    financial_calls: dict[str, int] | None = None
    read_only_ibkr_calls: dict[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "ibkr_contract_resolution_v1",
            "status": self.status.value,
            "request": self.request.as_dict(),
            "request_id": self.request_id,
            "source": self.source,
            "reason": self.reason,
            "returned_match_count": self.returned_match_count,
            "rejected_count": self.rejected_count,
            "cache_hit": self.cache_hit,
            "resolved_contract": self.resolved_contract_identity,
            "evaluation": self.evaluation,
            "persistence": self.persisted,
            "financial_calls": self.financial_calls or _zero_financial_calls(),
            "read_only_ibkr_calls": self.read_only_ibkr_calls or dict(READ_ONLY_CONTRACT_CALLS_ZERO),
        }


class LiveContractResolver:
    def __init__(
        self,
        service: ReadOnlyIbkrConnectionService,
        layout: ContractCacheLayout,
        *,
        request_ids: ContractRequestIdAllocator | None = None,
        timeout_seconds: float | None = None,
        provider_source: str = "ibkr_tws_paper",
    ) -> None:
        self.service = service
        self.layout = layout
        self.request_ids = request_ids or ContractRequestIdAllocator()
        self.timeout_seconds = timeout_seconds
        self.provider_source = provider_source

    def resolve(self, request: ContractResolutionRequest) -> LiveContractResolutionResult:
        try:
            request.validate_basic()
        except ValueError as exc:
            return self._error_result(
                request=request,
                status=ContractResolutionStatus.INVALID_REQUEST,
                source="validation",
                reason=str(exc),
            )

        initialize_contract_cache(self.layout)
        cache_validation = validate_contract_cache(self.layout)
        if cache_validation["status"] != "GO":
            return self._error_result(
                request=request,
                status=ContractResolutionStatus.PROVIDER_ERROR,
                source="local_contract_cache",
                reason="local contract cache validation failed before broker request",
                evaluation={"cache_validation": cache_validation},
            )

        try:
            cache_hit = find_fresh_contract_cache_hit(read_contract_cache_rows(self.layout), request)
        except ValueError as exc:
            return self._error_result(
                request=request,
                status=ContractResolutionStatus.PROVIDER_ERROR,
                source="local_contract_cache",
                reason=str(exc),
            )
        if cache_hit is not None:
            return LiveContractResolutionResult(
                status=ContractResolutionStatus.RESOLVED,
                request=request,
                request_id=None,
                source="local_contract_cache",
                reason="fresh exact contract cache hit",
                returned_match_count=1,
                rejected_count=0,
                cache_hit=True,
                resolved_contract_identity=contract_identity_document(cache_hit),
                financial_calls=_zero_financial_calls(),
                read_only_ibkr_calls=dict(READ_ONLY_CONTRACT_CALLS_ZERO),
            )

        snapshot = self.service.connect_once()
        if snapshot.status not in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}:
            self.service.disconnect()
            return self._error_result(
                request=request,
                status=ContractResolutionStatus.PROVIDER_ERROR,
                source=self.provider_source,
                reason=f"IBKR connection is not healthy: {snapshot.status.value}",
            )

        request_id = self.request_ids.next_id()
        event = self.service.state.contract_details_event(request_id)
        read_only_calls = dict(READ_ONLY_CONTRACT_CALLS_ZERO)
        try:
            spec = build_ibkr_contract_spec(request)
            self.service.app.reqContractDetails(  # type: ignore[union-attr]
                request_id,
                build_native_ibapi_contract(spec),
            )
            read_only_calls["req_contract_details"] = 1
            timeout = self.timeout_seconds or self.service.settings.request_timeout_seconds
            if not event.wait(timeout):
                return self._persisted_error_result(
                    request=request,
                    request_id=request_id,
                    status=ContractResolutionStatus.CALLBACK_TIMEOUT,
                    source=self.provider_source,
                    reason="contractDetailsEnd callback was not received before timeout",
                    read_only_calls=read_only_calls,
                )

            details = self.service.state.contract_details_for(request_id)
            evaluation = evaluate_contract_details_payloads(request, details)
            return self._result_from_evaluation(
                request=request,
                request_id=request_id,
                evaluation=evaluation,
                server_version=_server_version_as_int(snapshot.server_version),
                read_only_calls=read_only_calls,
            )
        except ValueError as exc:
            return self._persisted_error_result(
                request=request,
                request_id=request_id,
                status=ContractResolutionStatus.CONTRACT_VALIDATION_FAILED,
                source=self.provider_source,
                reason=str(exc),
                read_only_calls=read_only_calls,
            )
        except Exception as exc:
            return self._persisted_error_result(
                request=request,
                request_id=request_id,
                status=ContractResolutionStatus.PROVIDER_ERROR,
                source=self.provider_source,
                reason=f"{type(exc).__name__}: {exc}",
                read_only_calls=read_only_calls,
            )
        finally:
            self.service.disconnect()

    def _result_from_evaluation(
        self,
        *,
        request: ContractResolutionRequest,
        request_id: int,
        evaluation: ContractCandidateEvaluation,
        server_version: int | None,
        read_only_calls: dict[str, int],
    ) -> LiveContractResolutionResult:
        try:
            artifacts = build_contract_resolution_artifacts(
                request,
                evaluation,
                resolved_at=datetime.now(UTC),
                server_version=server_version,
            )
        except ValueError as exc:
            return self._persisted_error_result(
                request=request,
                request_id=request_id,
                status=ContractResolutionStatus.CONTRACT_VALIDATION_FAILED,
                source=self.provider_source,
                reason=str(exc),
                read_only_calls=read_only_calls,
                returned_match_count=len(evaluation.matches),
                rejected_count=evaluation.rejected_count,
                evaluation=evaluation.as_dict(),
            )

        persisted = persist_contract_resolution_artifacts(self.layout, artifacts)
        identity = artifacts.as_dict()["resolved_contract_identity"] if artifacts.cache_row else None
        return LiveContractResolutionResult(
            status=evaluation.status,
            request=request,
            request_id=request_id,
            source=self.provider_source,
            reason=evaluation.reason,
            returned_match_count=len(evaluation.matches),
            rejected_count=evaluation.rejected_count,
            cache_hit=False,
            persisted=persisted,
            resolved_contract_identity=identity,
            evaluation=evaluation.as_dict(),
            financial_calls=_zero_financial_calls(),
            read_only_ibkr_calls=read_only_calls,
        )

    def _persisted_error_result(
        self,
        *,
        request: ContractResolutionRequest,
        request_id: int,
        status: ContractResolutionStatus,
        source: str,
        reason: str,
        read_only_calls: dict[str, int],
        returned_match_count: int = 0,
        rejected_count: int = 0,
        evaluation: dict[str, Any] | None = None,
    ) -> LiveContractResolutionResult:
        error_evaluation = ContractCandidateEvaluation(
            status=status,
            matches=(),
            rejected_count=rejected_count,
            reason=reason,
        )
        record = build_resolution_audit_record(
            request,
            error_evaluation,
            resolved_at=datetime.now(UTC),
            server_version=None,
        )
        persisted = persist_contract_resolution_artifacts(
            self.layout,
            build_contract_resolution_artifacts(
                request,
                error_evaluation,
                resolved_at=datetime.now(UTC),
                server_version=None,
            ),
        )
        return LiveContractResolutionResult(
            status=status,
            request=request,
            request_id=request_id,
            source=source,
            reason=reason,
            returned_match_count=returned_match_count,
            rejected_count=rejected_count,
            cache_hit=False,
            persisted=persisted,
            evaluation=evaluation or {**error_evaluation.as_dict(), "audit_record": record},
            financial_calls=_zero_financial_calls(),
            read_only_ibkr_calls=read_only_calls,
        )

    @staticmethod
    def _error_result(
        *,
        request: ContractResolutionRequest,
        status: ContractResolutionStatus,
        source: str,
        reason: str,
        evaluation: dict[str, Any] | None = None,
    ) -> LiveContractResolutionResult:
        return LiveContractResolutionResult(
            status=status,
            request=request,
            request_id=None,
            source=source,
            reason=reason,
            returned_match_count=0,
            rejected_count=0,
            cache_hit=False,
            evaluation=evaluation,
            financial_calls=_zero_financial_calls(),
            read_only_ibkr_calls=dict(READ_ONLY_CONTRACT_CALLS_ZERO),
        )


def _zero_financial_calls() -> dict[str, int]:
    return {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }


def _server_version_as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
