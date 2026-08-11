from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stocks.capital.canary import (
    default_level_one_canary_policy,
    load_level_one_canary_policy,
)
from stocks.ibkr.p0_readiness import (
    P0_READINESS_BLOCKER,
    inspect_p0_execution_readiness_gate,
)
from stocks.research.autopilot.contracts import stable_hash


LIVE_LEVEL_ONE = "LIVE_LEVEL_ONE"
LIVE_LEVEL_TWO = "LIVE_LEVEL_TWO"
AUTONOMOUS_LEVEL_ONE = "AUTONOMOUS_LEVEL_ONE"
MANUAL_APPROVAL_ONLY = "MANUAL_APPROVAL_ONLY"
INACTIVE = "INACTIVE"
ACTIVE = "ACTIVE"
PAUSED = "PAUSED"
KILLED = "KILLED"
AUTONOMOUS_PROFILE = "autonomous_multi_asset_v1"
CAPABILITY_TTL_SECONDS = 900
LEVEL_TWO_ACTIVATION_APPROVAL = (
    "ACTIVATE LIVE LEVEL TWO WITH MANUAL APPROVAL"
)


def authority_status(project_root: Path) -> dict[str, Any]:
    state = _read_json(_state_path(project_root))
    allowlist = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "strategy-allowlist.json"
    )
    kill_switch = _read_json(
        project_root
        / "data"
        / "execution"
        / "live"
        / "private"
        / "kill-switch.json"
    )
    lifecycle_status = str(state.get("lifecycle_status", INACTIVE))
    state_authority = str(state.get("execution_authority", "NONE"))
    supported_authority = state_authority in {
        LIVE_LEVEL_ONE,
        AUTONOMOUS_LEVEL_ONE,
        LIVE_LEVEL_TWO,
    }
    qualification_hash_matches = bool(
        state.get("qualification_hash")
        and state.get("qualification_hash")
        == allowlist.get("qualification_hash")
    )
    allowlist_hash_matches = bool(
        state.get("allowlist_hash")
        and state.get("allowlist_hash")
        == operational_allowlist_hash(allowlist)
    )
    p0_gate = inspect_p0_execution_readiness_gate(project_root)
    p0_gate_go = p0_gate.get("status") == "GO"
    reconciliation_gate = _live_reconciliation_gate(
        project_root,
        expected_account_fingerprint_masked=state.get(
            "account_fingerprint_masked"
        ),
    )
    reconciliation_gate_go = reconciliation_gate.get("status") == "GO"
    capital_state = _read_json(
        project_root / "output" / "capital" / "current_level.json"
    )
    capital_level = int(
        capital_state.get("CURRENT_CAPITAL_LEVEL", 0) or 0
    )
    capital_level_matches = (
        state_authority != LIVE_LEVEL_TWO or capital_level >= 2
    )
    level_two_evidence = {"status": "NOT_REQUIRED", "blockers": []}
    if state_authority == LIVE_LEVEL_TWO:
        from stocks.live.evidence import live_level_two_evidence

        level_two_evidence = live_level_two_evidence(project_root)
    level_two_evidence_go = (
        state_authority != LIVE_LEVEL_TWO
        or level_two_evidence.get("status") == "GO"
    )
    autonomous_freeze = {"status": "NOT_REQUIRED", "blockers": []}
    if state_authority == AUTONOMOUS_LEVEL_ONE:
        from stocks.live.autonomous_policy import verify_p2_1_freeze

        autonomous_freeze = verify_p2_1_freeze(project_root)
    autonomous_freeze_go = (
        state_authority != AUTONOMOUS_LEVEL_ONE
        or (
            autonomous_freeze.get("status") == "GO"
            and state.get("p2_1_freeze_hash")
            == autonomous_freeze.get("freeze_hash")
        )
    )
    active = (
        lifecycle_status == ACTIVE
        and supported_authority
        and qualification_hash_matches
        and allowlist_hash_matches
        and p0_gate_go
        and reconciliation_gate_go
        and capital_level_matches
        and level_two_evidence_go
        and autonomous_freeze_go
        and not bool(kill_switch.get("active", False))
    )
    report = {
        "schema": "controlled_live_authority_status_v2",
        "status": "GO",
        "lifecycle_status": lifecycle_status,
        "execution_authority": state_authority if active else "NONE",
        "current_scaling_level": (
            "LEVEL_2"
            if active and state_authority == LIVE_LEVEL_TWO
            else "LEVEL_1"
            if active
            else "LEVEL_0"
        ),
        "activated_at": state.get("activated_at"),
        "paused_at": state.get("paused_at"),
        "killed_at": state.get("killed_at"),
        "qualification_hash_matches": qualification_hash_matches,
        "allowlist_hash_matches": allowlist_hash_matches,
        "p0_safety_gate_go": p0_gate_go,
        "p0_safety_attestation_hash": p0_gate.get("attestation_hash"),
        "p0_safety_blockers": p0_gate.get("blockers", []),
        "live_reconciliation_gate_go": reconciliation_gate_go,
        "live_reconciliation_blockers": reconciliation_gate.get(
            "blockers", []
        ),
        "capital_level": capital_level,
        "capital_level_matches_authority": capital_level_matches,
        "level_two_evidence_go": level_two_evidence_go,
        "level_two_evidence_hash": level_two_evidence.get("content_hash"),
        "level_two_evidence_blockers": level_two_evidence.get(
            "blockers", []
        ),
        "autonomous_freeze_go": autonomous_freeze_go,
        "p2_1_freeze_hash": state.get("p2_1_freeze_hash"),
        "current_p2_1_freeze_hash": autonomous_freeze.get("freeze_hash"),
        "autonomous_freeze_blockers": autonomous_freeze.get("blockers", []),
        "kill_switch_active": bool(kill_switch.get("active", False)),
        "account_fingerprint_masked": state.get(
            "account_fingerprint_masked"
        ),
        "limits": state.get("limits", _level_one_limits(project_root)),
        "active_strategies": state.get("active_strategies", []),
        "active_symbols": state.get("active_symbols", []),
        "submission_mode": state.get("submission_mode", "UNBOUND"),
        "manual_approval_required": state.get(
            "manual_approval_required", True
        ),
        "automatic_order_submission": state.get(
            "automatic_order_submission", False
        ),
        "whole_share_only": state.get("whole_share_only", True),
        "fractional_allowed": state.get("fractional_allowed", False),
        "risk_first": state.get("risk_first", True),
        "automatic_capital_promotion": False,
        "margin_enabled": False,
        "leverage_enabled": False,
        "shorting_enabled": False,
        "options_enabled": False,
        "futures_enabled": False,
    }
    _write_public(project_root, "authority-status.json", report)
    return report


def activate_level_one(
    project_root: Path,
    *,
    preflight: dict[str, Any],
    reauthorization: bool = False,
) -> dict[str, Any]:
    if preflight.get("status") != "GO" or preflight.get("blockers"):
        return _transition_blocked(
            project_root,
            "LIVE_PREFLIGHT_NOT_GO",
            list(preflight.get("blockers", [])),
        )
    p0_gate = inspect_p0_execution_readiness_gate(project_root)
    if p0_gate.get("status") != "GO":
        return _transition_blocked(
            project_root,
            P0_READINESS_BLOCKER,
            [P0_READINESS_BLOCKER, *p0_gate.get("blockers", [])],
        )
    reconciliation_gate = _live_reconciliation_gate(
        project_root,
        require_empty=True,
    )
    if reconciliation_gate.get("status") != "GO":
        return _transition_blocked(
            project_root,
            "LIVE_RECONCILIATION_GATE_REQUIRED",
            reconciliation_gate.get("blockers", []),
        )
    allowlist = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "strategy-allowlist.json"
    )
    strategies = list(allowlist.get("strategies", []))
    if allowlist.get("status") != "GO" or not strategies:
        return _transition_blocked(
            project_root,
            "PIT_STRATEGY_ALLOWLIST_REQUIRED",
            ["PIT_STRATEGY_ALLOWLIST_REQUIRED"],
        )
    reconciliation = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "reconciliation.json"
    )
    fingerprints = list(reconciliation.get("account_fingerprints", []))
    if len(fingerprints) != 1:
        return _transition_blocked(
            project_root,
            "LIVE_ACCOUNT_FINGERPRINT_MISMATCH",
            ["LIVE_ACCOUNT_FINGERPRINT_MISMATCH"],
        )
    prior = _read_json(_state_path(project_root))
    p02_freeze: dict[str, Any] = {}
    p02_policy: dict[str, Any] = {}
    if reauthorization:
        from stocks.live.level_one_reauthorization import (
            policy_integrity,
            verify_p02_freeze,
        )

        p02_freeze = verify_p02_freeze(project_root)
        p02_policy = policy_integrity(project_root)
        binding_blockers = [
            *(p02_freeze.get("blockers", []) if p02_freeze.get("status") != "GO" else []),
            *(p02_policy.get("blockers", []) if p02_policy.get("status") != "GO" else []),
        ]
        if binding_blockers:
            return _transition_blocked(
                project_root,
                "P02_REAUTHORIZATION_BINDING_BLOCKED",
                binding_blockers,
            )
    if str(prior.get("lifecycle_status")) == ACTIVE:
        current = authority_status(project_root)
        if (
            current["execution_authority"] == LIVE_LEVEL_ONE
            and not reauthorization
        ):
            return {
                **current,
                "transition_status": "ALREADY_ACTIVE_IDEMPOTENT",
            }
        if not reauthorization:
            return _transition_blocked(
                project_root,
                "ACTIVE_AUTHORITY_INTEGRITY_MISMATCH",
                ["ACTIVE_AUTHORITY_INTEGRITY_MISMATCH"],
            )
    now = _now()
    active_symbols = sorted(
        {
            str(symbol).upper()
            for strategy in strategies
            for symbol in strategy.get("allowed_symbols", [])
        }
    )
    state = {
        "schema": "live_level_one_private_authority_state_v2",
        "lifecycle_status": ACTIVE,
        "execution_authority": LIVE_LEVEL_ONE,
        "activated_at": now,
        "paused_at": None,
        "killed_at": None,
        "activation_id": f"L1-{uuid.uuid4().hex.upper()}",
        "reauthorization": reauthorization,
        "prior_authority_state_hash": stable_hash(prior) if prior else None,
        "p02_freeze_hash": p02_freeze.get("freeze_hash"),
        "whole_share_policy_hash": p02_policy.get("policy_hash"),
        "preflight_hash": stable_hash(preflight),
        "p0_safety_attestation_hash": p0_gate.get("attestation_hash"),
        "qualification_hash": allowlist.get("qualification_hash"),
        "allowlist_hash": operational_allowlist_hash(allowlist),
        "account_fingerprint_masked": _mask_fingerprint(fingerprints[0]),
        "active_strategies": [
            str(strategy["strategy_id"]) for strategy in strategies
        ],
        "active_symbols": active_symbols,
        "submission_mode": MANUAL_APPROVAL_ONLY,
        "manual_approval_required": True,
        "automatic_order_submission": False,
        "whole_share_only": True,
        "fractional_allowed": False,
        "risk_first": True,
        "canary_downscale_allowed": True,
        "limits": _level_one_limits(project_root),
    }
    _atomic_json(_state_path(project_root), state)
    _append_event(
        project_root,
        (
            "LIVE_LEVEL_ONE_REAUTHORIZED"
            if reauthorization
            else "LIVE_LEVEL_ONE_ACTIVATED"
        ),
        {
            "activation_id_hash": stable_hash(state["activation_id"]),
            "preflight_hash": state["preflight_hash"],
            "qualification_hash": state["qualification_hash"],
            "allowlist_hash": state["allowlist_hash"],
            "p02_freeze_hash": state["p02_freeze_hash"],
            "policy_hash": state["whole_share_policy_hash"],
        },
    )
    return {
        **authority_status(project_root),
        "transition_status": (
            "LIVE_LEVEL_ONE_REAUTHORIZED"
            if reauthorization
            else "LIVE_LEVEL_ONE_ACTIVATED"
        ),
    }


def activate_autonomous_level_one(
    project_root: Path,
    *,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Reauthorize Level-1 for machine policy decisions, never Level-2."""

    from stocks.live.autonomous_policy import (
        AUTONOMOUS_SUBMISSION_MODE,
        verify_p2_1_freeze,
    )

    freeze = verify_p2_1_freeze(project_root)
    if freeze.get("status") != "GO":
        return _transition_blocked(
            project_root,
            "P2_1_AUTONOMOUS_FREEZE_REQUIRED",
            list(freeze.get("blockers", [])),
        )
    transition = activate_level_one(
        project_root,
        preflight=preflight,
        reauthorization=True,
    )
    if transition.get("execution_authority") != LIVE_LEVEL_ONE:
        return transition
    prior = _read_json(_state_path(project_root))
    state = {
        **prior,
        "schema": "autonomous_level_one_private_authority_state_v1",
        "execution_authority": AUTONOMOUS_LEVEL_ONE,
        "submission_mode": AUTONOMOUS_SUBMISSION_MODE,
        "manual_approval_required": False,
        "automatic_order_submission": True,
        "autonomous_activated_at": _now(),
        "p2_1_freeze_hash": freeze.get("freeze_hash"),
        "per_trade_approval_required": False,
        "automatic_capital_promotion": False,
        "maximum_capital_level": 1,
    }
    _atomic_json(_state_path(project_root), state)
    _append_event(
        project_root,
        "AUTONOMOUS_LEVEL_ONE_ACTIVATED",
        {
            "activation_id_hash": stable_hash(state.get("activation_id")),
            "p2_1_freeze_hash": state["p2_1_freeze_hash"],
            "allowlist_hash": state.get("allowlist_hash"),
            "qualification_hash": state.get("qualification_hash"),
            "automatic_capital_promotion": False,
        },
    )
    return {
        **authority_status(project_root),
        "transition_status": "AUTONOMOUS_LEVEL_ONE_ACTIVATED",
        "p2_1_freeze_hash": state["p2_1_freeze_hash"],
        "state_changed": True,
    }


def activate_level_two(
    project_root: Path,
    *,
    symbol: str,
    approval: str,
    env_file: str | Path = ".env.ibkr.portfolio.live",
) -> dict[str, Any]:
    """Promote active Level-1 authority to bounded manual Level-2.

    The transition itself performs no broker operation.  It requires five
    ledger-bound closed Level-1 round trips, Level-2 capital, an executable
    desired target, a current writer freeze, P0 readiness and exact approval.
    """
    from stocks.capital.service import capital_command
    from stocks.live.evidence import live_level_two_evidence
    from stocks.live.portfolio_targets import controlled_live_preflight

    blockers: list[str] = []
    state = _read_json(_state_path(project_root))
    current = authority_status(project_root)
    resuming_level_two = bool(
        state.get("lifecycle_status") == PAUSED
        and state.get("paused_execution_authority") == LIVE_LEVEL_TWO
    )
    if not resuming_level_two and (
        state.get("lifecycle_status") != ACTIVE
        or state.get("execution_authority") != LIVE_LEVEL_ONE
        or current.get("execution_authority") != LIVE_LEVEL_ONE
    ):
        blockers.append("ACTIVE_LIVE_LEVEL_ONE_AUTHORITY_REQUIRED")

    capital = capital_command(project_root, "status")
    if int(capital.get("CURRENT_CAPITAL_LEVEL", 0) or 0) < 2:
        blockers.append("CAPITAL_LEVEL_2_REQUIRED")
    evidence = live_level_two_evidence(project_root)
    if evidence.get("status") != "GO":
        blockers.extend(evidence.get("blockers", []))
        blockers.append("VERIFIED_LEVEL_ONE_ROUND_TRIPS_REQUIRED")
    preflight = controlled_live_preflight(
        project_root,
        symbol=symbol,
        env_file=env_file,
        require_authority=False,
    )
    if preflight.get("status") != "GO":
        blockers.extend(preflight.get("blockers", []))
        blockers.append("CONTROLLED_LEVEL_TWO_PREFLIGHT_REQUIRED")
    if approval != LEVEL_TWO_ACTIVATION_APPROVAL:
        blockers.append("EXACT_LEVEL_TWO_ACTIVATION_APPROVAL_REQUIRED")
    if blockers:
        report = _transition_blocked(
            project_root,
            "LIVE_LEVEL_TWO_ACTIVATION_BLOCKED",
            blockers,
        )
        report["approval_challenge"] = LEVEL_TWO_ACTIVATION_APPROVAL
        report["required_symbol"] = symbol.upper()
        _write_public(project_root, "authority-transition.json", report)
        return report

    safe_config = dict(preflight.get("safe_config", {}))
    state.update(
        schema="controlled_live_private_authority_state_v2",
        lifecycle_status=ACTIVE,
        execution_authority=LIVE_LEVEL_TWO,
        level_two_activated_at=_now(),
        level_two_activation_id=f"L2-{uuid.uuid4().hex.upper()}",
        level_two_preflight_hash=stable_hash(preflight),
        level_two_evidence_hash=evidence.get("content_hash"),
        level_two_capital_state_hash=stable_hash(capital),
        limits=_level_two_limits(safe_config),
        paused_at=None,
        pause_reason=None,
        paused_execution_authority=None,
    )
    _atomic_json(_state_path(project_root), state)
    _append_event(
        project_root,
        "LIVE_LEVEL_TWO_ACTIVATED",
        {
            "activation_id_hash": stable_hash(
                state["level_two_activation_id"]
            ),
            "preflight_hash": state["level_two_preflight_hash"],
            "evidence_hash": state["level_two_evidence_hash"],
            "capital_state_hash": state["level_two_capital_state_hash"],
            "symbol": symbol.upper(),
        },
    )
    return {
        **authority_status(project_root),
        "transition_status": "LIVE_LEVEL_TWO_ACTIVATED",
        "state_changed": True,
        "broker_writes": 0,
    }


def create_live_capability(
    project_root: Path,
    *,
    preflight: dict[str, Any],
    confirmed: bool,
    profile: str = AUTONOMOUS_PROFILE,
    now: datetime | None = None,
) -> dict[str, Any]:
    blockers = []
    if not confirmed:
        blockers.append("EXPLICIT_YES_CONFIRMATION_REQUIRED")
    if profile != AUTONOMOUS_PROFILE:
        blockers.append("UNKNOWN_LIVE_PROFILE")
    profile_config = _profile_config(project_root, profile)
    if not _valid_profile_config(profile_config, profile):
        blockers.append("LIVE_PROFILE_CONFIG_INVALID")
    if preflight.get("status") != "GO":
        blockers.extend(preflight.get("blockers", []))
        blockers.append("LIVE_PREFLIGHT_NOT_GO")
    p0_gate = inspect_p0_execution_readiness_gate(project_root)
    if p0_gate.get("status") != "GO":
        blockers.extend(
            [P0_READINESS_BLOCKER, *p0_gate.get("blockers", [])]
        )
    reconciliation_gate = _live_reconciliation_gate(
        project_root,
        require_empty=True,
    )
    if reconciliation_gate.get("status") != "GO":
        blockers.extend(reconciliation_gate.get("blockers", []))
    if blockers:
        return _capability_blocked(project_root, blockers)
    created = now or datetime.now(UTC)
    binding = _capability_binding(project_root, preflight, profile)
    if not binding.get("account_fingerprint_masked"):
        return _capability_blocked(
            project_root,
            ["LIVE_ACCOUNT_FINGERPRINT_MISMATCH"],
        )
    capability_id = f"CAP-{uuid.uuid4().hex.upper()}"
    private = {
        "schema": "controlled_live_capability_private_v1",
        "status": "READY",
        "capability_id": capability_id,
        "profile": profile,
        "created_at": created.isoformat(),
        "expires_at": (
            created + timedelta(seconds=CAPABILITY_TTL_SECONDS)
        ).isoformat(),
        "consumed_at": None,
        "binding": binding,
        "binding_hash": stable_hash(binding),
        "execution_authority": "NONE",
    }
    _atomic_json(_capability_path(project_root), private)
    _append_event(
        project_root,
        "LIVE_CAPABILITY_CREATED",
        {
            "capability_id_hash": stable_hash(capability_id),
            "binding_hash": private["binding_hash"],
        },
    )
    return _publish_capability(project_root, private)


def live_capability_status(
    project_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    private = _read_json(_capability_path(project_root))
    if not private:
        return _capability_blocked(
            project_root,
            ["LIVE_CAPABILITY_NOT_CREATED"],
        )
    current = now or datetime.now(UTC)
    expires_at = _parse_datetime(private.get("expires_at"))
    current_binding = _capability_binding(
        project_root,
        {
            "safe_config": private.get("binding", {}).get(
                "safe_config",
                {},
            )
        },
        str(private.get("profile", "")),
    )
    blockers = []
    if private.get("status") != "READY":
        blockers.append("LIVE_CAPABILITY_NOT_READY")
    if private.get("consumed_at"):
        blockers.append("LIVE_CAPABILITY_ALREADY_CONSUMED")
    if expires_at is None or current >= expires_at:
        blockers.append("LIVE_CAPABILITY_EXPIRED")
    if private.get("binding_hash") != stable_hash(current_binding):
        blockers.append("LIVE_CAPABILITY_BINDING_CHANGED")
    if blockers:
        return _capability_blocked(project_root, blockers, private=private)
    return _publish_capability(project_root, private)


def activate_live_capability(
    project_root: Path,
    *,
    preflight: dict[str, Any],
    confirmed: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not confirmed:
        return _transition_blocked(
            project_root,
            "EXPLICIT_YES_CONFIRMATION_REQUIRED",
            ["EXPLICIT_YES_CONFIRMATION_REQUIRED"],
        )
    if preflight.get("status") != "GO" or preflight.get("blockers"):
        return _transition_blocked(
            project_root,
            "LIVE_PREFLIGHT_NOT_GO",
            list(preflight.get("blockers", [])),
        )
    capability = live_capability_status(project_root, now=now)
    if capability.get("status") != "GO":
        return _transition_blocked(
            project_root,
            "LIVE_CAPABILITY_NOT_READY",
            list(capability.get("blockers", [])),
        )
    transition = activate_level_one(project_root, preflight=preflight)
    if transition.get("execution_authority") != LIVE_LEVEL_ONE:
        return transition
    private = _read_json(_capability_path(project_root))
    private.update(
        status="CONSUMED",
        consumed_at=(now or datetime.now(UTC)).isoformat(),
        execution_authority=LIVE_LEVEL_ONE,
        submission_mode=MANUAL_APPROVAL_ONLY,
        manual_approval_required=True,
        automatic_order_submission=False,
        whole_share_only=True,
        fractional_allowed=False,
        risk_first=True,
    )
    _atomic_json(_capability_path(project_root), private)
    _append_event(
        project_root,
        "LIVE_CAPABILITY_CONSUMED",
        {
            "capability_id_hash": stable_hash(
                str(private.get("capability_id", ""))
            ),
            "binding_hash": private.get("binding_hash"),
        },
    )
    return {
        **transition,
        "capability_status": "CONSUMED",
        "profile": private.get("profile"),
    }


def pause_level_one(
    project_root: Path, *, reason: str
) -> dict[str, Any]:
    state = _read_json(_state_path(project_root))
    if str(state.get("lifecycle_status")) != ACTIVE:
        return _transition_blocked(
            project_root,
            "LIVE_LEVEL_ONE_NOT_ACTIVE",
            ["LIVE_LEVEL_ONE_NOT_ACTIVE"],
        )
    prior_authority = str(state.get("execution_authority") or "NONE")
    state.update(
        lifecycle_status=PAUSED,
        execution_authority="NONE",
        paused_execution_authority=prior_authority,
        paused_at=_now(),
        pause_reason=reason.strip() or "OPERATOR_PAUSE",
    )
    _atomic_json(_state_path(project_root), state)
    _append_event(
        project_root,
        "LIVE_LEVEL_ONE_PAUSED",
        {"reason_hash": stable_hash(state["pause_reason"])},
    )
    return {
        **authority_status(project_root),
        "transition_status": "LIVE_LEVEL_ONE_PAUSED",
    }


def resume_level_one(
    project_root: Path,
    *,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    state = _read_json(_state_path(project_root))
    if str(state.get("lifecycle_status")) != PAUSED:
        return _transition_blocked(
            project_root,
            "LIVE_LEVEL_ONE_NOT_PAUSED",
            ["LIVE_LEVEL_ONE_NOT_PAUSED"],
        )
    if state.get("paused_execution_authority") == LIVE_LEVEL_TWO:
        return _transition_blocked(
            project_root,
            "LIVE_LEVEL_TWO_REACTIVATION_REQUIRES_EXPLICIT_APPROVAL",
            ["LIVE_LEVEL_TWO_REACTIVATION_REQUIRES_EXPLICIT_APPROVAL"],
        )
    if preflight.get("status") != "GO" or preflight.get("blockers"):
        return _transition_blocked(
            project_root,
            "LIVE_RESUME_PREFLIGHT_NOT_GO",
            list(preflight.get("blockers", [])),
        )
    p0_gate = inspect_p0_execution_readiness_gate(project_root)
    if p0_gate.get("status") != "GO":
        return _transition_blocked(
            project_root,
            P0_READINESS_BLOCKER,
            [P0_READINESS_BLOCKER, *p0_gate.get("blockers", [])],
        )
    reconciliation_gate = _live_reconciliation_gate(
        project_root,
        expected_account_fingerprint_masked=state.get(
            "account_fingerprint_masked"
        ),
        require_empty=True,
    )
    if reconciliation_gate.get("status") != "GO":
        return _transition_blocked(
            project_root,
            "LIVE_RECONCILIATION_GATE_REQUIRED",
            reconciliation_gate.get("blockers", []),
        )
    allowlist = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "strategy-allowlist.json"
    )
    if (
        state.get("qualification_hash") != allowlist.get("qualification_hash")
        or state.get("allowlist_hash")
        != operational_allowlist_hash(allowlist)
    ):
        return _transition_blocked(
            project_root,
            "FROZEN_LIVE_ALLOWLIST_CHANGED",
            ["FROZEN_LIVE_ALLOWLIST_CHANGED"],
        )
    resumed_authority = str(
        state.get("paused_execution_authority") or LIVE_LEVEL_ONE
    )
    if resumed_authority == AUTONOMOUS_LEVEL_ONE:
        from stocks.live.autonomous_policy import verify_p2_1_freeze

        autonomous_freeze = verify_p2_1_freeze(project_root)
        if autonomous_freeze.get("status") != "GO":
            return _transition_blocked(
                project_root,
                "P2_1_AUTONOMOUS_FREEZE_REQUIRED",
                list(autonomous_freeze.get("blockers", [])),
            )
    state.update(
        lifecycle_status=ACTIVE,
        execution_authority=resumed_authority,
        paused_at=None,
        pause_reason=None,
        resumed_at=_now(),
        resume_preflight_hash=stable_hash(preflight),
        p0_safety_attestation_hash=p0_gate.get("attestation_hash"),
    )
    _atomic_json(_state_path(project_root), state)
    _append_event(
        project_root,
        "LIVE_LEVEL_ONE_RESUMED",
        {"preflight_hash": state["resume_preflight_hash"]},
    )
    return {
        **authority_status(project_root),
        "transition_status": (
            "AUTONOMOUS_LEVEL_ONE_RESUMED"
            if resumed_authority == AUTONOMOUS_LEVEL_ONE
            else "LIVE_LEVEL_ONE_RESUMED"
        ),
    }


def kill_level_one(
    project_root: Path, *, reason: str
) -> dict[str, Any]:
    if not reason.strip():
        return _transition_blocked(
            project_root,
            "KILL_REASON_REQUIRED",
            ["KILL_REASON_REQUIRED"],
        )
    state = _read_json(_state_path(project_root))
    state.update(
        schema="live_level_one_private_authority_state_v1",
        lifecycle_status=KILLED,
        execution_authority="NONE",
        killed_at=_now(),
        kill_reason=reason.strip(),
    )
    _atomic_json(_state_path(project_root), state)
    _append_event(
        project_root,
        "LIVE_LEVEL_ONE_KILLED",
        {"reason_hash": stable_hash(reason.strip())},
    )
    return {
        **authority_status(project_root),
        "transition_status": "LIVE_LEVEL_ONE_KILLED",
    }


def _transition_blocked(
    project_root: Path, reason: str, blockers: list[str]
) -> dict[str, Any]:
    report = {
        "schema": "live_level_one_authority_transition_v1",
        "status": "NO_GO",
        "transition_status": reason,
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
        "state_changed": False,
        "broker_writes": 0,
    }
    _write_public(project_root, "authority-transition.json", report)
    return report


def _level_one_limits(project_root: Path) -> dict[str, Any]:
    try:
        policy = load_level_one_canary_policy(project_root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        policy = default_level_one_canary_policy()
    return {
        "capital_level_name": "WHOLE_SHARE_EXECUTION_CANARY",
        "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
        "canary_notional_hard_cap_eur": str(
            policy.hard_notional_cap_eur
        ),
        "notional_cap_role": "SECONDARY_EMERGENCY_BACKSTOP",
        "max_total_live_exposure_eur": str(
            policy.hard_notional_cap_eur
        ),
        "canary_risk_pct": str(policy.canary_risk_pct),
        "maximum_canary_risk_eur": str(policy.maximum_risk_eur),
        "maximum_stock_weight": str(policy.maximum_stock_weight),
        "maximum_pooled_vehicle_weight": str(
            policy.maximum_pooled_vehicle_weight
        ),
        "max_open_live_positions": 1,
        "max_new_live_orders_per_day": 1,
        "maximum_quantity": policy.maximum_execution_quantity,
        "whole_shares_only": True,
        "fractional_shares_allowed": False,
        "autoscaling": False,
    }


def _level_two_limits(safe_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_order_value_eur": safe_config.get("max_order_eur"),
        "max_total_live_exposure_eur": safe_config.get(
            "max_total_exposure_eur"
        ),
        "max_risk_per_trade_eur": safe_config.get("max_risk_eur"),
        "max_open_live_positions": safe_config.get("max_open_positions"),
        "max_new_live_orders_per_day": safe_config.get(
            "max_new_orders_per_day"
        ),
        "maximum_quantity": safe_config.get("maximum_quantity"),
        "whole_shares_only": True,
        "per_order_manual_approval": True,
        "automatic_orders": False,
        "autoscaling": False,
        "margin_enabled": False,
        "leverage_enabled": False,
        "shorting_enabled": False,
    }


def _mask_fingerprint(value: Any) -> str:
    return f"fingerprint-sha256:{stable_hash(str(value))[:16].lower()}"


def operational_allowlist_hash(allowlist: dict[str, Any]) -> str:
    strategies = []
    for strategy in allowlist.get("strategies", []):
        if not isinstance(strategy, dict):
            continue
        strategies.append(
            {
                key: strategy.get(key)
                for key in (
                    "strategy_id",
                    "version",
                    "source_hash",
                    "parameter_hash",
                    "qualification_hash",
                    "training_data_end",
                    "allowed_timeframes",
                    "allowed_asset_classes",
                    "allowed_symbols",
                    "maximum_position_weight",
                    "canary_notional_hard_cap_eur",
                    "primary_sizing_authority",
                    "maximum_daily_orders",
                    "maximum_daily_loss_eur",
                    "status",
                )
            }
        )
    strategies.sort(key=lambda row: str(row.get("strategy_id", "")))
    return stable_hash(
        {
            "schema": allowlist.get("schema"),
            "status": allowlist.get("status"),
            "qualification_hash": allowlist.get("qualification_hash"),
            "strategies": strategies,
        }
    )


def _capability_binding(
    project_root: Path,
    preflight: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    allowlist = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "strategy-allowlist.json"
    )
    reconciliation = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "reconciliation.json"
    )
    fingerprints = list(reconciliation.get("account_fingerprints", []))
    safe_config = dict(preflight.get("safe_config", {}))
    profile_config = _profile_config(project_root, profile)
    p0_gate = inspect_p0_execution_readiness_gate(project_root)
    return {
        "profile": profile,
        "profile_config_hash": (
            stable_hash(profile_config) if profile_config else None
        ),
        "p0_safety_status": p0_gate.get("status"),
        "p0_safety_attestation_hash": p0_gate.get("attestation_hash"),
        "qualification_hash": allowlist.get("qualification_hash"),
        "allowlist_hash": operational_allowlist_hash(allowlist),
        "strategy_ids": sorted(
            str(row.get("strategy_id"))
            for row in allowlist.get("strategies", [])
            if row.get("strategy_id")
        ),
        "symbols": sorted(
            {
                str(symbol).upper()
                for row in allowlist.get("strategies", [])
                for symbol in row.get("allowed_symbols", [])
            }
        ),
        "account_fingerprint_masked": (
            _mask_fingerprint(fingerprints[0])
            if len(fingerprints) == 1
            else None
        ),
        "reconciliation_status": reconciliation.get(
            "reconciliation_status"
        ),
        "reconciliation_state": {
            key: reconciliation.get(key)
            for key in (
                "status",
                "reconciliation_status",
                "position_count",
                "open_order_count",
                "same_client_open_order_count",
                "all_api_open_order_count",
                "unknown_positions",
                "unknown_orders",
                "account_fingerprint_count",
            )
        },
        "safe_config": {
            key: safe_config.get(key)
            for key in (
                "environment",
                "host_local_only",
                "port_class",
                "configured_notional_backstop_eur",
                "primary_sizing_authority",
                "notional_cap_role",
                "canary_risk_cap_eur",
                "max_total_exposure_eur",
                "max_open_positions",
                "max_new_orders_per_day",
                "autoscaling_enabled",
                "futures_allowed",
            )
        },
        "limits": _level_one_limits(project_root),
    }


def _live_reconciliation_gate(
    project_root: Path,
    *,
    expected_account_fingerprint_masked: Any = None,
    require_empty: bool = False,
) -> dict[str, Any]:
    reconciliation = _read_json(
        project_root
        / "output"
        / "ibkr"
        / "live"
        / "reconciliation.json"
    )
    blockers: list[str] = []
    observed_hash = reconciliation.get("content_hash")
    expected_hash = stable_hash(
        {
            key: value
            for key, value in reconciliation.items()
            if key != "content_hash"
        }
    )
    if not observed_hash or observed_hash != expected_hash:
        blockers.append("LIVE_RECONCILIATION_INTEGRITY_BLOCKED")
    reconciliation_status = str(
        reconciliation.get("reconciliation_status", "")
    )
    if (
        reconciliation.get("status") != "GO"
        or not reconciliation_status.startswith("LIVE_RECONCILED")
    ):
        blockers.append("LIVE_RECONCILIATION_NOT_GO")
    if require_empty and reconciliation_status != "LIVE_RECONCILED_EMPTY":
        blockers.append("LIVE_RECONCILIATION_EMPTY_NOT_PROVEN")
    if reconciliation.get("unknown_orders") != 0:
        blockers.append("LIVE_UNKNOWN_ORDERS_BLOCKED")
    if reconciliation.get("unknown_positions") != 0:
        blockers.append("LIVE_UNKNOWN_POSITIONS_BLOCKED")
    fingerprints = list(reconciliation.get("account_fingerprints", []))
    if len(fingerprints) != 1:
        blockers.append("LIVE_ACCOUNT_FINGERPRINT_MISMATCH")
    elif (
        expected_account_fingerprint_masked
        and _mask_fingerprint(fingerprints[0])
        != expected_account_fingerprint_masked
    ):
        blockers.append("LIVE_ACCOUNT_FINGERPRINT_CHANGED")
    return {
        "status": "GO" if not blockers else "NO_GO",
        "blockers": sorted(set(blockers)),
        "reconciliation_status": reconciliation_status,
        "content_hash": observed_hash,
    }


def _publish_capability(
    project_root: Path,
    private: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "schema": "controlled_live_capability_public_v1",
        "status": "GO",
        "capability_status": private.get("status"),
        "profile": private.get("profile"),
        "created_at": private.get("created_at"),
        "expires_at": private.get("expires_at"),
        "binding_hash": private.get("binding_hash"),
        "strategy_count": len(
            private.get("binding", {}).get("strategy_ids", [])
        ),
        "symbol_count": len(
            private.get("binding", {}).get("symbols", [])
        ),
        "account_fingerprint_masked": private.get("binding", {}).get(
            "account_fingerprint_masked"
        ),
        "execution_authority": (
            LIVE_LEVEL_ONE
            if private.get("status") == "CONSUMED"
            else "NONE"
        ),
        "broker_writes": 0,
        "blockers": [],
    }
    _write_public(project_root, "capability.json", report)
    return report


def _capability_blocked(
    project_root: Path,
    blockers: list[str],
    *,
    private: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema": "controlled_live_capability_public_v1",
        "status": "NO_GO",
        "capability_status": (
            private.get("status", "BLOCKED")
            if private
            else "BLOCKED"
        ),
        "profile": (
            private.get("profile") if private else AUTONOMOUS_PROFILE
        ),
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
        "broker_writes": 0,
    }
    _write_public(project_root, "capability.json", report)
    return report


def _capability_path(project_root: Path) -> Path:
    return _state_path(project_root).with_name("capability.json")


def _profile_config(project_root: Path, profile: str) -> dict[str, Any]:
    return _read_json(
        project_root
        / "config"
        / "operations"
        / f"{profile}.json"
    )


def _valid_profile_config(
    config: dict[str, Any],
    profile: str,
) -> bool:
    authority = config.get("authority", {})
    constraints = config.get("constraints", {})
    return bool(
        config.get("schema") == "stocks_operations_profile_v1"
        and config.get("profile_id") == profile
        and config.get("execution_mode") == "CONTROLLED_LIVE"
        and authority.get("initial_operator_activation_required") is True
        and authority.get("per_order_approval_required") is False
        and constraints.get("margin_enabled") is False
        and constraints.get("leverage_enabled") is False
        and constraints.get("shorting_enabled") is False
        and constraints.get("futures_enabled") is False
    )


def _parse_datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _state_path(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "execution"
        / "live"
        / "private"
        / "authority-state.json"
    )


def _append_event(
    project_root: Path, event_type: str, payload: dict[str, Any]
) -> None:
    path = _state_path(project_root).with_name("authority-events.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "schema": "live_level_one_authority_event_v1",
        "event_id": f"AUTH-{uuid.uuid4().hex.upper()}",
        "event_type": event_type,
        "created_at": _now(),
        "payload": payload,
    }
    event["content_hash"] = stable_hash(event)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _write_public(
    project_root: Path, name: str, payload: dict[str, Any]
) -> None:
    _atomic_json(
        project_root / "output" / "ibkr" / "live" / name,
        payload,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _now() -> str:
    return datetime.now(UTC).isoformat()
