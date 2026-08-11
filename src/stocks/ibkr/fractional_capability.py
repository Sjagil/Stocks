from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Protocol

from stocks.execution.idempotency import stable_hash


OUTPUT_PATH = Path("output/ibkr/capabilities/fractional-shares.json")
_UNSET_DECIMAL_THRESHOLD = Decimal("1000000000")
_INFORMATIONAL_ERROR_CODES = {2104, 2106, 2107, 2108, 2158}


@dataclass(frozen=True)
class FractionalProbeSettings:
    host: str
    port: int
    client_id: int
    timeout_seconds: float = 10.0

    def blockers(self, *, reserved_client_ids: set[int] | None = None) -> list[str]:
        blockers: list[str] = []
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            blockers.append("NON_LOCAL_IBKR_HOST_BLOCKED")
        if self.port not in {7496, 7497, 4001, 4002}:
            blockers.append("UNRECOGNIZED_IBKR_PORT")
        if self.client_id <= 0:
            blockers.append("DEDICATED_PROBE_CLIENT_ID_REQUIRED")
        if reserved_client_ids and self.client_id in reserved_client_ids:
            blockers.append("PROBE_CLIENT_ID_COLLIDES_WITH_CONFIGURED_CLIENT")
        if not 0.1 <= self.timeout_seconds <= 60.0:
            blockers.append("PROBE_TIMEOUT_OUT_OF_BOUNDS")
        return blockers


class FractionalProbeApp(Protocol):
    details: list[Any]
    errors: list[dict[str, Any]]

    def connect(self, host: str, port: int, client_id: int) -> None: ...

    def is_connected(self) -> bool: ...

    def start(self) -> None: ...

    def wait_ready(self, timeout_seconds: float) -> bool: ...

    def request_contract_details(self, request_id: int, contract: Any) -> None: ...

    def wait_complete(self, timeout_seconds: float) -> bool: ...

    def server_version(self) -> int | None: ...

    def close(self) -> None: ...


class NativeFractionalProbeApp:
    def __init__(self) -> None:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        outer = self

        class _App(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                outer.ready.set()

            def contractDetails(self, reqId: int, details: Any) -> None:  # noqa: N802
                if int(reqId) == outer.request_id:
                    outer.details.append(details)

            def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
                if int(reqId) == outer.request_id:
                    outer.complete.set()

            def error(self, reqId: Any, *args: Any) -> None:  # type: ignore[override]
                code, message = _parse_error_args(args)
                outer.errors.append(
                    {
                        "request_id": str(reqId),
                        "code": code,
                        "message_hash": stable_hash(message),
                        "informational": code in _INFORMATIONAL_ERROR_CODES,
                    }
                )

            def connectionClosed(self) -> None:  # noqa: N802
                outer.connection_closed = True

        self._app = _App()
        self.ready = threading.Event()
        self.complete = threading.Event()
        self.details: list[Any] = []
        self.errors: list[dict[str, Any]] = []
        self.request_id = -1
        self.connection_closed = False
        self.thread: threading.Thread | None = None

    def connect(self, host: str, port: int, client_id: int) -> None:
        self._app.connect(host, port, client_id)

    def is_connected(self) -> bool:
        return bool(self._app.isConnected())

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._app.run,
            name="ibkr-read-only-fractional-capability-probe",
            daemon=True,
        )
        self.thread.start()

    def wait_ready(self, timeout_seconds: float) -> bool:
        return self.ready.wait(timeout_seconds)

    def request_contract_details(self, request_id: int, contract: Any) -> None:
        self.request_id = request_id
        self._app.reqContractDetails(request_id, contract)

    def wait_complete(self, timeout_seconds: float) -> bool:
        return self.complete.wait(timeout_seconds)

    def server_version(self) -> int | None:
        value = _integer(self._app.serverVersion())
        return value if value and value > 0 else None

    def close(self) -> None:
        if self._app.isConnected():
            self._app.disconnect()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def probe_fractional_contract_capability(
    project_root: Path,
    *,
    settings: FractionalProbeSettings,
    con_id: int,
    symbol: str,
    currency: str,
    exchange: str = "SMART",
    reserved_client_ids: set[int] | None = None,
    app_factory: Callable[[], FractionalProbeApp] = NativeFractionalProbeApp,
) -> dict[str, Any]:
    blockers = settings.blockers(reserved_client_ids=reserved_client_ids)
    if con_id <= 0:
        blockers.append("POSITIVE_CON_ID_REQUIRED")
    if not symbol.strip():
        blockers.append("SYMBOL_REQUIRED")
    if exchange.upper() != "SMART":
        blockers.append("FRACTIONAL_PROBE_REQUIRES_SMART_ROUTING")
    if blockers:
        return _publish(
            project_root,
            _base_report(
                status="PROBE_PREFLIGHT_BLOCKED",
                settings=settings,
                con_id=con_id,
                symbol=symbol,
                currency=currency,
                exchange=exchange,
                blockers=blockers,
            ),
        )

    app = app_factory()
    request_count = 0
    try:
        app.connect(settings.host, settings.port, settings.client_id)
        if not app.is_connected():
            return _publish(
                project_root,
                _base_report(
                    status="PROBE_CONNECTION_BLOCKED",
                    settings=settings,
                    con_id=con_id,
                    symbol=symbol,
                    currency=currency,
                    exchange=exchange,
                    blockers=["IBKR_SOCKET_NOT_CONNECTED"],
                ),
            )
        app.start()
        if not app.wait_ready(settings.timeout_seconds):
            return _publish(
                project_root,
                _base_report(
                    status="PROBE_HANDSHAKE_TIMEOUT",
                    settings=settings,
                    con_id=con_id,
                    symbol=symbol,
                    currency=currency,
                    exchange=exchange,
                    blockers=["NEXT_VALID_ID_CALLBACK_TIMEOUT"],
                ),
            )
        request_id = 9_301_001
        app.request_contract_details(
            request_id,
            _stock_contract(con_id, symbol, currency, exchange),
        )
        request_count = 1
        if not app.wait_complete(settings.timeout_seconds):
            return _publish(
                project_root,
                _base_report(
                    status="PROBE_CALLBACK_TIMEOUT",
                    settings=settings,
                    con_id=con_id,
                    symbol=symbol,
                    currency=currency,
                    exchange=exchange,
                    blockers=["CONTRACT_DETAILS_END_TIMEOUT"],
                    request_count=request_count,
                    errors=app.errors,
                    server_version=app.server_version(),
                ),
            )
        report = _classify_details(
            settings=settings,
            con_id=con_id,
            symbol=symbol,
            currency=currency,
            exchange=exchange,
            details=app.details,
            errors=app.errors,
            server_version=app.server_version(),
            request_count=request_count,
        )
        return _publish(project_root, report)
    except Exception as exc:
        return _publish(
            project_root,
            _base_report(
                status="PROBE_PROVIDER_ERROR",
                settings=settings,
                con_id=con_id,
                symbol=symbol,
                currency=currency,
                exchange=exchange,
                blockers=[f"{type(exc).__name__}_DURING_READ_ONLY_PROBE"],
                request_count=request_count,
                errors=getattr(app, "errors", []),
            ),
        )
    finally:
        app.close()


def _classify_details(
    *,
    settings: FractionalProbeSettings,
    con_id: int,
    symbol: str,
    currency: str,
    exchange: str,
    details: list[Any],
    errors: list[dict[str, Any]],
    server_version: int | None,
    request_count: int,
) -> dict[str, Any]:
    matching = [
        row
        for row in details
        if _integer(getattr(getattr(row, "contract", None), "conId", None))
        == con_id
    ]
    if len(matching) != 1:
        return _base_report(
            status=(
                "CONTRACT_NOT_FOUND" if not matching else "AMBIGUOUS_CONTRACT_BLOCKED"
            ),
            settings=settings,
            con_id=con_id,
            symbol=symbol,
            currency=currency,
            exchange=exchange,
            blockers=[
                "EXACT_CONTRACT_DETAIL_REQUIRED"
                if not matching
                else "MULTIPLE_EXACT_CONTRACT_DETAILS_BLOCKED"
            ],
            request_count=request_count,
            errors=errors,
            server_version=server_version,
            match_count=len(matching),
        )
    row = matching[0]
    observed = {
        "min_size": _usable_decimal(getattr(row, "minSize", None)),
        "size_increment": _usable_decimal(getattr(row, "sizeIncrement", None)),
        "suggested_size_increment": _usable_decimal(
            getattr(row, "suggestedSizeIncrement", None)
        ),
    }
    usable = [value for value in observed.values() if value is not None]
    if not usable:
        classification = "CONTRACT_FRACTIONAL_CAPABILITY_UNPROVEN"
        status = "UNPROVEN"
        blockers = ["CONTRACT_SIZE_INCREMENT_METADATA_UNAVAILABLE"]
    elif any(value < Decimal("1") for value in usable):
        classification = "CONTRACT_FRACTIONAL_INCREMENT_OBSERVED"
        status = "GO_CONTRACT_METADATA_ONLY"
        blockers = [
            "ACCOUNT_FRACTIONAL_PERMISSION_UNPROVEN",
            "FRACTIONAL_ORDER_TYPE_SUPPORT_UNPROVEN",
            "FRACTIONAL_BRACKET_SUPPORT_UNPROVEN",
        ]
    else:
        classification = "CONTRACT_WHOLE_SHARE_ONLY_OBSERVED"
        status = "NO_GO"
        blockers = ["CONTRACT_METADATA_DOES_NOT_EXPOSE_FRACTIONAL_INCREMENT"]
    report = _base_report(
        status=status,
        settings=settings,
        con_id=con_id,
        symbol=symbol,
        currency=currency,
        exchange=exchange,
        blockers=blockers,
        request_count=request_count,
        errors=errors,
        server_version=server_version,
        match_count=1,
    )
    report.update(
        {
            "classification": classification,
            "contract_size_metadata": {
                key: None if value is None else str(value)
                for key, value in observed.items()
            },
            "contract_fractional_increment_observed": (
                classification == "CONTRACT_FRACTIONAL_INCREMENT_OBSERVED"
            ),
            "account_fractional_permission_proven": False,
            "fractional_order_type_support_proven": False,
            "fractional_bracket_support_proven": False,
            "fractional_writer_activation_allowed": False,
        }
    )
    return report


def _base_report(
    *,
    status: str,
    settings: FractionalProbeSettings,
    con_id: int,
    symbol: str,
    currency: str,
    exchange: str,
    blockers: list[str],
    request_count: int = 0,
    errors: list[dict[str, Any]] | None = None,
    server_version: int | None = None,
    match_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema": "ibkr_fractional_contract_capability_v1",
        "status": status,
        "observed_at": datetime.now(UTC).isoformat(),
        "contract_reference": {
            "con_id_hash": stable_hash(str(con_id)),
            "symbol": symbol.upper(),
            "security_type": "STK",
            "currency": currency.upper(),
            "exchange": exchange.upper(),
        },
        "contract_match_count": match_count,
        "server_version": server_version,
        "read_only_request_counters": {"contract_details_requests": request_count},
        "forbidden_write_counters": {
            "place_order_calls": 0,
            "cancel_order_calls": 0,
            "global_cancel_calls": 0,
            "request_order_id_calls": 0,
            "auto_bind_order_calls": 0,
            "exercise_option_calls": 0,
        },
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "errors": errors or [],
        "error_count": len(errors or []),
        "blockers": sorted(set(blockers)),
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
        "account_ids_stored": 0,
        "credentials_stored": 0,
        "dedicated_client_id_used": settings.client_id > 0,
    }


def _publish(project_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report["content_hash"] = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _stock_contract(
    con_id: int, symbol: str, currency: str, exchange: str
) -> Any:
    from ibapi.contract import Contract

    contract = Contract()
    contract.conId = con_id
    contract.symbol = symbol.upper()
    contract.secType = "STK"
    contract.exchange = exchange.upper()
    contract.currency = currency.upper()
    return contract


def _usable_decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not parsed.is_finite() or parsed <= 0 or parsed >= _UNSET_DECIMAL_THRESHOLD:
        return None
    return parsed


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_error_args(args: tuple[Any, ...]) -> tuple[int | None, str]:
    # IBAPI 10.48 inserts errorTime before errorCode; older clients do not.
    if len(args) >= 3:
        return _integer(args[1]), str(args[2])[:300]
    if len(args) >= 2:
        return _integer(args[0]), str(args[1])[:300]
    return None, str(args)[:300]


__all__ = [
    "FractionalProbeSettings",
    "NativeFractionalProbeApp",
    "probe_fractional_contract_capability",
]
