from __future__ import annotations

from enum import StrEnum


class ExecutionAuthority(StrEnum):
    NONE = "NONE"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE = "LIVE"


def phase7_authority_status(authority: ExecutionAuthority | str) -> dict[str, object]:
    value = ExecutionAuthority(str(authority))
    if value is ExecutionAuthority.NONE:
        return {"status": "GO", "execution_authority": value.value, "decision_code": "AUTHORITY_NONE_SIMULATION_ONLY"}
    return {"status": "NO_GO", "execution_authority": value.value, "decision_code": "AUTHORITY_NOT_GRANTED"}

