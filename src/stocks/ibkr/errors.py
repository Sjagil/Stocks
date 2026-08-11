from __future__ import annotations

from typing import Any


INFO_CODES = {2104, 2106, 2107, 2108, 2158}
DEGRADED_CODES = {2103, 2105, 2157}
FATAL_CODES = {326, 502, 504, 1100}


def to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize_error(req_id_or_exception: Any, callback_args: tuple[Any, ...]) -> dict[str, Any]:
    if isinstance(req_id_or_exception, BaseException) and not callback_args:
        return {
            "req_id": -1,
            "error_time": None,
            "code": "CLIENT_EXCEPTION",
            "message": to_text(req_id_or_exception),
            "advanced_order_reject_json_present": False,
        }

    req_id = (
        int(req_id_or_exception)
        if isinstance(req_id_or_exception, (int, float))
        else -1
    )
    error_time: int | None = None
    error_code: int | str = "UNKNOWN_ERROR"
    error_string: Any = ""
    advanced_reject: Any = ""

    if len(callback_args) == 2:
        error_code, error_string = callback_args
    elif len(callback_args) == 3:
        first, second, third = callback_args
        if isinstance(second, (int, float)):
            error_time = int(first) if isinstance(first, (int, float)) else None
            error_code = int(second)
            error_string = third
        else:
            error_code = int(first) if isinstance(first, (int, float)) else first
            error_string = second
            advanced_reject = third
    elif len(callback_args) >= 4:
        raw_time, raw_code, error_string, advanced_reject = callback_args[:4]
        error_time = int(raw_time) if isinstance(raw_time, (int, float)) else None
        error_code = int(raw_code) if isinstance(raw_code, (int, float)) else raw_code
    else:
        return {
            "req_id": req_id,
            "error_time": None,
            "code": "UNEXPECTED_ERROR_CALLBACK",
            "message": f"Unexpected callback arguments: {callback_args!r}",
            "advanced_order_reject_json_present": False,
        }

    return {
        "req_id": req_id,
        "error_time": error_time,
        "code": error_code,
        "message": to_text(error_string),
        "advanced_order_reject_json_present": bool(advanced_reject),
    }


def is_info_code(code: Any) -> bool:
    return isinstance(code, int) and code in INFO_CODES


def is_degraded_code(code: Any) -> bool:
    return isinstance(code, int) and code in DEGRADED_CODES


def is_fatal_code(code: Any) -> bool:
    return code == "CLIENT_EXCEPTION" or (isinstance(code, int) and code in FATAL_CODES)
