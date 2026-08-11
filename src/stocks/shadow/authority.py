from __future__ import annotations

from enum import Enum

from stocks.shadow.errors import AUTHORITY_NOT_GRANTED


class StrategyAuthority(str, Enum):
    NONE = "NONE"
    RESEARCH_SHADOW = "RESEARCH_SHADOW"
    PAPER = "PAPER"
    LIMITED_LIVE = "LIMITED_LIVE"
    LIVE = "LIVE"


class ShadowAuthority(str, Enum):
    NONE = "NONE"
    RESEARCH_SHADOW = "RESEARCH_SHADOW"
    PAPER = "PAPER"
    LIMITED_LIVE = "LIMITED_LIVE"
    LIVE = "LIVE"


class ExecutionAuthority(str, Enum):
    NONE = "NONE"
    RESEARCH_SHADOW = "RESEARCH_SHADOW"
    PAPER = "PAPER"
    LIMITED_LIVE = "LIMITED_LIVE"
    LIVE = "LIVE"


def authority_status(value: str | StrategyAuthority | ShadowAuthority | ExecutionAuthority) -> dict[str, str]:
    raw = value.value if isinstance(value, Enum) else str(value)
    if raw == "NONE":
        return {"status": "GO", "authority": "NONE", "decision_code": "AUTHORITY_NONE_CONFIRMED"}
    return {"status": "NO_GO", "authority": raw, "decision_code": AUTHORITY_NOT_GRANTED}


def authority_contract() -> dict[str, str]:
    return {
        "strategy_authority": "NONE",
        "shadow_authority": "NONE",
        "execution_authority": "NONE",
    }
