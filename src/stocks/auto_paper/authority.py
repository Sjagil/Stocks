from __future__ import annotations

from dataclasses import dataclass

from stocks.auto_paper.config import AutoPaperConfig


@dataclass(frozen=True)
class AuthorityDependencies:
    phase9_full_frozen_go: bool
    financial_finalist_go: bool
    forward_shadow_go: bool
    paper_account_verified: bool
    account_fingerprint_match: bool
    reconciliation_go: bool
    shariah_fresh: bool = False
    signal_fresh: bool = False
    quote_fresh: bool = False
    risk_go: bool = False


def entry_authority(config: AutoPaperConfig, dependencies: AuthorityDependencies, strategy_id: str) -> dict[str, object]:
    blockers = []
    if not config.enabled:
        blockers.append("AUTO_PAPER_DISABLED")
    if not dependencies.phase9_full_frozen_go:
        blockers.append("PHASE9_FULL_FREEZE_REQUIRED")
    if not dependencies.financial_finalist_go:
        blockers.append("FINANCIAL_FINALIST_REQUIRED")
    if not dependencies.forward_shadow_go:
        blockers.append("FORWARD_SHADOW_GO_REQUIRED")
    if not dependencies.paper_account_verified:
        blockers.append("PAPER_ACCOUNT_NOT_VERIFIED")
    if not dependencies.account_fingerprint_match:
        blockers.append("ACCOUNT_FINGERPRINT_MISMATCH")
    if not dependencies.reconciliation_go:
        blockers.append("BROKER_RECONCILIATION_MISMATCH")
    if not dependencies.shariah_fresh:
        blockers.append("SHARIAH_STATUS_STALE")
    if not dependencies.signal_fresh:
        blockers.append("STALE_SIGNAL")
    if not dependencies.quote_fresh:
        blockers.append("STALE_QUOTE")
    if not dependencies.risk_go:
        blockers.append("RISK_GATE_BLOCKED")
    if strategy_id not in config.strategy_allowlist:
        blockers.append("STRATEGY_NOT_ALLOWLISTED")
    return {
        "status": "GO" if not blockers else "BLOCKED",
        "authority_type": "AUTOMATED_PAPER_ENTRY",
        "entry_authority": "AUTOMATED_PAPER_ENTRY" if not blockers else "NONE",
        "execution_authority": "AUTOMATED_PAPER_ENTRY" if not blockers else "NONE",
        "automatic_submission": not blockers,
        "blockers": blockers,
    }


def risk_reducing_exit_authority(
    *,
    existing_long_position: bool,
    quantities_match: bool,
    account_match: bool,
    con_id_match: bool,
    sell_within_position: bool,
    limit_day_rth: bool,
    runtime_enabled: bool,
) -> dict[str, object]:
    blockers = []
    if not existing_long_position:
        blockers.append("SELL_WITHOUT_POSITION_BLOCKED")
    if not quantities_match or not account_match or not con_id_match:
        blockers.append("EXIT_BLOCKED_POSITION_MISMATCH")
    if not sell_within_position:
        blockers.append("SELL_EXCEEDS_RECONCILED_POSITION")
    if not limit_day_rth:
        blockers.append("EXIT_ORDER_CONTRACT_BLOCKED")
    technically_eligible = not blockers
    runtime_go = technically_eligible and runtime_enabled
    return {
        "status": "GO" if runtime_go else "BLOCKED",
        "technical_status": "GO" if technically_eligible else "BLOCKED",
        "authority_type": "AUTOMATED_PAPER_RISK_REDUCING_EXIT",
        "runtime_authority": "AUTOMATED_PAPER_RISK_REDUCING_EXIT" if runtime_go else "NONE",
        "automatic_submission": runtime_go,
        "blockers": blockers + ([] if runtime_enabled else ["AUTO_PAPER_DISABLED"]),
    }


def foundation_authority() -> dict[str, object]:
    return {
        "AUTO_PAPER_TECHNICAL_READINESS": "GO",
        "AUTO_PAPER_RUNTIME_AUTHORITY": "BLOCKED",
        "execution_authority": "NONE",
        "entry_authority": "NONE",
        "risk_reducing_exit_authority": "NONE",
        "strategy_authority": "NONE",
        "live_authority": "NONE",
        "automatic_submission": False,
        "runtime_blockers": [
            "PHASE9_FULL_FREEZE_REQUIRED",
            "FINANCIAL_FINALIST_REQUIRED",
            "FORWARD_SHADOW_GO_REQUIRED",
        ],
    }
