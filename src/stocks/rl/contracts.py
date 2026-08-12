from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from typing import Any, Mapping, Sequence


class RLAction(IntEnum):
    HOLD = 0
    OPEN_SMALL = 1
    OPEN_NORMAL = 2
    OPEN_LARGE = 3
    REDUCE_25 = 4
    REDUCE_50 = 5
    CLOSE = 6
    TIGHTEN_STOP = 7


ACTION_NAMES = {int(action): action.name for action in RLAction}
OPEN_ACTIONS = frozenset(
    {RLAction.OPEN_SMALL, RLAction.OPEN_NORMAL, RLAction.OPEN_LARGE}
)
MANAGEMENT_ACTIONS = frozenset(
    {RLAction.REDUCE_25, RLAction.REDUCE_50, RLAction.CLOSE, RLAction.TIGHTEN_STOP}
)


class RLMode(StrEnum):
    SHADOW_ONLY = "SHADOW_ONLY"
    PAPER = "PAPER"
    LIVE = "LIVE"


class PolicyState(StrEnum):
    ACTIVE = "ACTIVE"
    CHALLENGER = "CHALLENGER"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


class RLEvidenceStatus(StrEnum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    SHADOW_ONLY = "SHADOW_ONLY"
    REJECTED = "REJECTED"
    PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"


@dataclass(frozen=True)
class RLRuntimeConfig:
    mode: RLMode = RLMode.SHADOW_ONLY
    live_enabled: bool = False
    broker_execution_authority: str = "NONE"
    policy_may_control_money: bool = False
    fallback: str = "NO_RL_ACTION"
    inference_interval_seconds: int = 900
    evaluation_interval_hours: int = 24
    retraining_interval_hours: int = 24
    minimum_new_closed_episodes: int = 100

    def validate(self) -> None:
        if self.inference_interval_seconds < 60:
            raise ValueError("RL inference interval must be at least 60 seconds")
        if self.evaluation_interval_hours < 1 or self.retraining_interval_hours < 1:
            raise ValueError("RL evaluation and retraining intervals must be positive")
        if self.minimum_new_closed_episodes < 1:
            raise ValueError("minimum_new_closed_episodes must be positive")
        if self.policy_may_control_money:
            raise ValueError("RL policy money control is forbidden")
        if self.mode is RLMode.LIVE:
            if not self.live_enabled:
                raise ValueError("RL live mode requires explicit RL_LIVE_ENABLED")
            if self.broker_execution_authority == "NONE":
                raise ValueError("RL live mode requires external broker authority")
        if self.fallback not in {"NO_RL_ACTION", "DETERMINISTIC_ENGINE"}:
            raise ValueError("unsupported RL fallback")

    def public_contract(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "mode": self.mode.value,
            "rl_has_direct_broker_authority": False,
            "rl_may_override_risk": False,
            "rl_may_override_shariah": False,
            "rl_may_self_promote": False,
            "execution_authority": "NONE",
            "money_control": False,
        }


@dataclass(frozen=True)
class HardGates:
    risk_budget: bool = True
    shariah: bool = True
    tradeable: bool = True
    liquidity: bool = True
    portfolio_capacity: bool = True
    data_fresh: bool = True
    quote_valid: bool = True
    execution_authorized: bool = False

    def entry_allowed(self, *, require_execution_authority: bool) -> bool:
        checks = (
            self.risk_budget,
            self.shariah,
            self.tradeable,
            self.liquidity,
            self.portfolio_capacity,
            self.data_fresh,
            self.quote_valid,
        )
        return all(checks) and (
            self.execution_authorized or not require_execution_authority
        )

    def failed(self, *, require_execution_authority: bool) -> list[str]:
        values = {
            "RISK_BUDGET": self.risk_budget,
            "SHARIAH": self.shariah,
            "TRADEABLE": self.tradeable,
            "LIQUIDITY": self.liquidity,
            "PORTFOLIO_CAPACITY": self.portfolio_capacity,
            "DATA_FRESH": self.data_fresh,
            "LIVE_QUOTE": self.quote_valid,
        }
        if require_execution_authority:
            values["EXECUTION_AUTHORITY"] = self.execution_authorized
        return [name for name, passed in values.items() if not passed]


@dataclass(frozen=True)
class PositionState:
    weight: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    peak_price: float = 0.0
    trough_price: float = 0.0
    holding_steps: int = 0

    @property
    def is_open(self) -> bool:
        return self.weight > 1e-12 and self.entry_price > 0


def action_mask(
    position: PositionState,
    gates: HardGates,
    *,
    require_execution_authority: bool = False,
    stop_can_tighten: bool = True,
) -> tuple[bool, ...]:
    """Return the only actions an RL proposal may select.

    Historical/shadow counterfactual environments deliberately do not require
    broker authority because proposals are never submitted. Live execution
    callers must set ``require_execution_authority=True`` and still pass the
    proposal through the canonical deterministic execution writer.
    """

    mask = [False] * len(RLAction)
    mask[RLAction.HOLD] = True
    if position.is_open:
        mask[RLAction.REDUCE_25] = True
        mask[RLAction.REDUCE_50] = True
        mask[RLAction.CLOSE] = True
        mask[RLAction.TIGHTEN_STOP] = bool(stop_can_tighten)
    elif gates.entry_allowed(
        require_execution_authority=require_execution_authority
    ):
        mask[RLAction.OPEN_SMALL] = True
        mask[RLAction.OPEN_NORMAL] = True
        mask[RLAction.OPEN_LARGE] = True
    return tuple(mask)


def validate_observation(values: Sequence[float]) -> None:
    if not values:
        raise ValueError("RL observation is empty")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("RL observation contains non-finite values")


def stable_hash(value: Mapping[str, Any] | Sequence[Any] | str) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def action_name(action: int | RLAction) -> str:
    try:
        return RLAction(int(action)).name
    except (TypeError, ValueError):
        return "INVALID"


def decision_type_for_action(
    action: int | RLAction, *, position_open: bool
) -> str:
    try:
        selected = RLAction(int(action))
    except (TypeError, ValueError):
        return "INVALID_DECISION"
    if selected in OPEN_ACTIONS:
        return "ENTRY_DECISION"
    if selected is RLAction.HOLD:
        return "HOLD_DECISION" if position_open else "SKIP_DECISION"
    if selected is RLAction.CLOSE:
        return "EXIT_DECISION"
    return "MANAGEMENT_DECISION"
