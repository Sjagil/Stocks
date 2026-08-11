from __future__ import annotations

from enum import StrEnum

from stocks.ibkr.paper_execution.errors import AUTHORITY_NOT_GRANTED


class Phase9ExecutionAuthority(StrEnum):
    NONE = "NONE"
    MANUAL_PAPER_CANARY = "MANUAL_PAPER_CANARY"
    PAPER = "PAPER"
    LIVE = "LIVE"


def phase9_authority_status(authority: str | Phase9ExecutionAuthority) -> dict[str, object]:
    value = str(authority)
    if value in {Phase9ExecutionAuthority.NONE.value, Phase9ExecutionAuthority.MANUAL_PAPER_CANARY.value}:
        return {"status": "GO", "execution_authority": value}
    return {"status": "NO_GO", "execution_authority": value, "decision_code": AUTHORITY_NOT_GRANTED}


def authority_contract(*, enabled: bool = False) -> dict[str, object]:
    return {
        "execution_authority": "MANUAL_PAPER_CANARY" if enabled else "NONE",
        "strategy_authority": "NONE",
        "shadow_authority": "NONE",
        "live_authority": "NONE",
        "manual_approval_required": True,
        "automatic_submission": False,
        "paper_only": True,
    }
