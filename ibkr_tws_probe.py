from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


PAPER_PORTS = {7497, 4002}
LIVE_PORTS = {7496, 4001}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
INFO_CODES = {2104, 2106, 2107, 2108, 2158}


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {_to_text(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _to_text(value)


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _masked_account_count(accounts_csv: str | bytes) -> int:
    text = _to_text(accounts_csv)
    return len([item for item in text.split(",") if item.strip()])


@dataclass
class ProbeResult:
    status: str = "IBKR_TWS_PROBE_NO_GO"
    host: str = ""
    port: int = 0
    client_id: int = 0
    connected: bool = False
    api_ready: bool = False
    event_loop_alive: bool = False
    server_version: int | str | None = None
    connection_time: str | None = None
    ibkr_unix_time: int | None = None
    ibkr_utc_time: str | None = None
    managed_account_count: int = 0
    ibapi_distribution_version: str | None = None
    ibapi_module_path: str | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    informational_messages: list[dict[str, Any]] = field(default_factory=list)
    financial_calls: dict[str, int] = field(
        default_factory=lambda: {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        }
    )

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema": "ibkr_tws_read_only_probe_v3",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                **asdict(self),
            }
        )


def _load_env(env_file: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "python-dotenv ontbreekt. Installeer eerst requirements.txt."
        ) from exc

    if not env_file.exists():
        raise SystemExit(f"Env-bestand niet gevonden: {env_file}")

    load_dotenv(env_file, override=False)


def _print_result(result: ProbeResult) -> None:
    print(
        json.dumps(
            result.as_dict(),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def _parse_error_callback(
    req_id_or_exception: Any,
    callback_args: tuple[Any, ...],
) -> dict[str, Any]:
    """
    Normalize both legacy and IBKR API 10.48+ EWrapper.error signatures.

    Supported shapes include:
      error(exception)
      error(reqId, errorCode, errorString)
      error(reqId, errorCode, errorString, advancedRejectJson)
      error(reqId, errorTime, errorCode, errorString)
      error(reqId, errorTime, errorCode, errorString, advancedRejectJson)
    """
    if isinstance(req_id_or_exception, BaseException) and not callback_args:
        return {
            "req_id": -1,
            "error_time": None,
            "code": "CLIENT_EXCEPTION",
            "message": _to_text(req_id_or_exception),
            "advanced_order_reject_json_present": False,
        }

    req_id = int(req_id_or_exception) if isinstance(
        req_id_or_exception, (int, float)
    ) else -1
    error_time: int | None = None
    error_code: int | str = "UNKNOWN_ERROR"
    error_string = ""
    advanced_reject: Any = ""

    if len(callback_args) == 2:
        # Legacy: error(reqId, errorCode, errorString)
        error_code, error_string = callback_args

    elif len(callback_args) == 3:
        first, second, third = callback_args

        if isinstance(second, (int, float)):
            # New: error(reqId, errorTime, errorCode, errorString)
            error_time = int(first) if isinstance(first, (int, float)) else None
            error_code = int(second)
            error_string = third
        else:
            # Legacy: error(reqId, errorCode, errorString, advancedRejectJson)
            error_code = int(first) if isinstance(first, (int, float)) else first
            error_string = second
            advanced_reject = third

    elif len(callback_args) >= 4:
        # New: error(reqId, errorTime, errorCode, errorString, advancedRejectJson)
        raw_time, raw_code, error_string, advanced_reject = callback_args[:4]
        error_time = int(raw_time) if isinstance(raw_time, (int, float)) else None
        error_code = int(raw_code) if isinstance(raw_code, (int, float)) else raw_code

    else:
        return {
            "req_id": req_id,
            "error_time": None,
            "code": "UNEXPECTED_ERROR_CALLBACK",
            "message": f"Onverwachte callbackargumenten: {callback_args!r}",
            "advanced_order_reject_json_present": False,
        }

    return {
        "req_id": req_id,
        "error_time": error_time,
        "code": error_code,
        "message": _to_text(error_string),
        "advanced_order_reject_json_present": bool(advanced_reject),
    }


def run_probe(env_file: Path) -> int:
    _load_env(env_file)

    host = os.getenv("IBKR_HOST", "127.0.0.1").strip()
    port = int(os.getenv("IBKR_PORT", "7497"))
    client_id = int(os.getenv("IBKR_CLIENT_ID", "17"))
    live_enabled = _as_bool(
        os.getenv("IBKR_LIVE_TRADING_ENABLED"),
        default=False,
    )
    allow_transmission = _as_bool(
        os.getenv("IBKR_ALLOW_ORDER_TRANSMISSION"),
        default=False,
    )
    authority = os.getenv("IBKR_ORDER_AUTHORITY", "NONE").strip().upper()
    connect_timeout = float(
        os.getenv("IBKR_CONNECT_TIMEOUT_SECONDS", "12")
    )

    result = ProbeResult(
        host=host,
        port=port,
        client_id=client_id,
    )

    if host not in LOCAL_HOSTS:
        result.errors.append(
            {
                "code": "NON_LOCAL_HOST_BLOCKED",
                "message": host,
            }
        )
        _print_result(result)
        return 2

    if port in LIVE_PORTS or port not in PAPER_PORTS:
        result.errors.append(
            {
                "code": "NON_PAPER_PORT_BLOCKED",
                "message": (
                    f"Fase 0 staat alleen poorten "
                    f"{sorted(PAPER_PORTS)} toe."
                ),
            }
        )
        _print_result(result)
        return 2

    if live_enabled or allow_transmission or authority != "NONE":
        result.errors.append(
            {
                "code": "AUTHORITY_NOT_READ_ONLY",
                "message": (
                    "IBKR_LIVE_TRADING_ENABLED=false, "
                    "IBKR_ALLOW_ORDER_TRANSMISSION=false en "
                    "IBKR_ORDER_AUTHORITY=NONE zijn verplicht."
                ),
            }
        )
        _print_result(result)
        return 2

    try:
        import ibapi
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
    except ModuleNotFoundError:
        result.errors.append(
            {
                "code": "IBAPI_NOT_INSTALLED",
                "message": (
                    "Installeer de officiële IBKR Python API uit "
                    r"C:\TWS API\source\pythonclient."
                ),
            }
        )
        _print_result(result)
        return 2

    result.ibapi_distribution_version = _distribution_version("ibapi")
    result.ibapi_module_path = str(Path(ibapi.__file__).resolve())

    ready = threading.Event()
    clock_received = threading.Event()
    accounts_received = threading.Event()

    class ReadOnlyProbe(EWrapper, EClient):
        def __init__(self) -> None:
            EClient.__init__(self, self)

        def nextValidId(self, orderId: int) -> None:  # noqa: N802
            result.api_ready = True
            ready.set()
            self.reqCurrentTime()
            self.reqManagedAccts()

        def currentTime(self, unix_time: int) -> None:  # noqa: N802
            result.ibkr_unix_time = int(unix_time)
            result.ibkr_utc_time = datetime.fromtimestamp(
                unix_time,
                tz=timezone.utc,
            ).isoformat()
            clock_received.set()

        def managedAccounts(  # noqa: N802
            self,
            accountsList: str | bytes,
        ) -> None:
            result.managed_account_count = _masked_account_count(
                accountsList
            )
            accounts_received.set()

        def error(  # type: ignore[override]
            self,
            reqId: Any,
            *args: Any,
        ) -> None:
            normalized = _parse_error_callback(reqId, args)
            code = normalized["code"]

            if isinstance(code, int) and code in INFO_CODES:
                result.informational_messages.append(normalized)
                return

            result.errors.append(normalized)

        def connectionClosed(self) -> None:  # noqa: N802
            result.connected = False

    app = ReadOnlyProbe()
    event_thread: threading.Thread | None = None
    return_code = 2

    try:
        app.connect(host, port, client_id)

        if not app.isConnected():
            result.errors.append(
                {
                    "code": "SOCKET_NOT_CONNECTED",
                    "message": (
                        "TWS accepteerde geen socketverbinding. Controleer "
                        "paperlogin, API sockets en poort 7497."
                    ),
                }
            )
            return 2

        result.connected = True
        result.server_version = _json_safe(app.serverVersion())
        result.connection_time = _to_text(app.twsConnectionTime())

        event_thread = threading.Thread(
            target=app.run,
            name="ibkr-read-only-probe-v3",
            daemon=True,
        )
        event_thread.start()

        if not ready.wait(connect_timeout):
            result.errors.append(
                {
                    "code": "API_HANDSHAKE_TIMEOUT",
                    "message": (
                        "Socket geopend, maar geen nextValidId ontvangen."
                    ),
                }
            )
        else:
            clock_received.wait(connect_timeout)
            accounts_received.wait(connect_timeout)

            result.event_loop_alive = event_thread.is_alive()

            if not result.event_loop_alive:
                result.errors.append(
                    {
                        "code": "EVENT_LOOP_TERMINATED",
                        "message": (
                            "De IBKR eventthread stopte onverwacht."
                        ),
                    }
                )
            elif not clock_received.is_set():
                result.errors.append(
                    {
                        "code": "CURRENT_TIME_TIMEOUT",
                        "message": (
                            "De API-handshake werkte, maar reqCurrentTime "
                            "gaf geen antwoord binnen de timeout."
                        ),
                    }
                )
            elif not accounts_received.is_set():
                result.errors.append(
                    {
                        "code": "MANAGED_ACCOUNTS_TIMEOUT",
                        "message": (
                            "De serverklok werd ontvangen, maar "
                            "managedAccounts niet binnen de timeout."
                        ),
                    }
                )
            else:
                result.status = "IBKR_TWS_READ_ONLY_PROBE_GO"
                return_code = 0

    except Exception as exc:
        result.errors.append(
            {
                "code": type(exc).__name__,
                "message": _to_text(exc),
            }
        )
    finally:
        try:
            app.disconnect()
        except Exception as exc:
            result.errors.append(
                {
                    "code": "DISCONNECT_ERROR",
                    "message": _to_text(exc),
                }
            )

        time.sleep(0.15)
        result.connected = False
        if event_thread is not None:
            result.event_loop_alive = event_thread.is_alive()

        _print_result(result)

    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed read-only connectivity probe voor IBKR TWS paper. "
            "Compatibel met legacy error callbacks en API 10.48+."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env.ibkr"),
        help="Pad naar het veilige IBKR env-bestand.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_probe(args.env_file)


if __name__ == "__main__":
    raise SystemExit(main())
