from __future__ import annotations

import hmac
import re
from hashlib import sha256
from typing import Any


RAW_ACCOUNT_PATTERN = re.compile(
    r"\b(?:DU\d{4,}|U\d{4,}|DU_TEST_ACCOUNT_\d+)\b"
)


def account_fingerprint(raw_account_id: str, key: str) -> str:
    if not key:
        raise ValueError("ACCOUNT_FINGERPRINT_KEY_MISSING")
    if not raw_account_id:
        raise ValueError("ACCOUNT_MASKING_FAILURE")
    return hmac.new(key.encode("utf-8"), raw_account_id.encode("utf-8"), sha256).hexdigest()


def hash_optional_text(value: str | bytes | None, key: str) -> str | None:
    if value is None:
        return None
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    if text == "":
        return None
    return hmac.new(key.encode("utf-8"), text.encode("utf-8"), sha256).hexdigest()


def contains_raw_account(value: Any) -> bool:
    return bool(RAW_ACCOUNT_PATTERN.search(_stringify(value)))


def scrub_raw_accounts(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: scrub_raw_accounts(item) for key, item in value.items() if key not in {"account", "raw_account_id"}}
    if isinstance(value, list):
        return [scrub_raw_accounts(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_raw_accounts(item) for item in value)
    if isinstance(value, str):
        return RAW_ACCOUNT_PATTERN.sub("ACCOUNT_MASKED", value)
    return value


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return repr(value)
    return str(value)
