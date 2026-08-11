from __future__ import annotations

import re


_CURRENCY_CODE_PATTERN = re.compile(r"^[A-Z]{3}$")


def validate_currency_code(value: str) -> None:
    if not _CURRENCY_CODE_PATTERN.fullmatch(value.strip()):
        raise ValueError("currency must be a 3-letter uppercase ISO-style code")
