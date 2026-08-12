from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping, Sequence

import pandas as pd

from stocks.data.multitimeframe import CANONICAL_INTERVALS, canonical_interval
from stocks.execution.idempotency import stable_hash


class SwingLifecycleState(StrEnum):
    DISCOVERED = "DISCOVERED"
    WATCHING = "WATCHING"
    NEAR_SETUP = "NEAR_SETUP"
    SETUP_VALID = "SETUP_VALID"
    ENTRY_READY = "ENTRY_READY"
    ENTERED = "ENTERED"
    POSITION_ACTIVE = "POSITION_ACTIVE"
    PROFIT_PROTECTION = "PROFIT_PROTECTION"
    REDUCE_CANDIDATE = "REDUCE_CANDIDATE"
    ROTATION_CANDIDATE = "ROTATION_CANDIDATE"
    EXIT_READY = "EXIT_READY"
    CLOSED = "CLOSED"
    INVALIDATED = "INVALIDATED"


@dataclass(frozen=True)
class StrategyTimeframeContract:
    entry_timeframe: str
    setup_timeframe: str
    context_timeframes: tuple[str, ...]
    structural_timeframe: str
    management_timeframe: str
    exit_timeframe: str
    required_timeframes: tuple[str, ...]
    optional_timeframes: tuple[str, ...] = ()
    session: str = "RTH"

    def __post_init__(self) -> None:
        scalar_fields = (
            "entry_timeframe",
            "setup_timeframe",
            "structural_timeframe",
            "management_timeframe",
            "exit_timeframe",
        )
        for name in scalar_fields:
            object.__setattr__(self, name, canonical_interval(getattr(self, name)))
        object.__setattr__(
            self,
            "context_timeframes",
            _canonical_unique(self.context_timeframes),
        )
        object.__setattr__(
            self,
            "required_timeframes",
            _canonical_unique(self.required_timeframes),
        )
        object.__setattr__(
            self,
            "optional_timeframes",
            _canonical_unique(self.optional_timeframes),
        )
        required = set(self.required_timeframes)
        optional = set(self.optional_timeframes)
        if required & optional:
            raise ValueError("required and optional timeframes must be disjoint")
        if self.entry_timeframe not in required:
            raise ValueError("entry_timeframe must be required")
        if self.setup_timeframe not in required:
            raise ValueError("setup_timeframe must be required")
        declared = required | optional
        used = {
            self.structural_timeframe,
            self.management_timeframe,
            self.exit_timeframe,
            *self.context_timeframes,
        }
        if not used.issubset(declared):
            missing = sorted(used - declared, key=_timeframe_order)
            raise ValueError(f"role timeframes must be declared: {missing}")
        if str(self.session).upper() not in {"RTH", "EXTENDED", "ALL"}:
            raise ValueError("session must be RTH, EXTENDED, or ALL")
        object.__setattr__(self, "session", str(self.session).upper())

    @property
    def all_timeframes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {*self.required_timeframes, *self.optional_timeframes},
                key=_timeframe_order,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "active_swing_strategy_timeframe_contract_v1",
            **asdict(self),
            "context_timeframes": list(self.context_timeframes),
            "required_timeframes": list(self.required_timeframes),
            "optional_timeframes": list(self.optional_timeframes),
            "earliest_core_decision_timeframe": "15m",
            "all_timeframes_need_not_agree": True,
        }


@dataclass(frozen=True)
class ActiveSwingOpportunity:
    symbol: str
    security_id: str
    strategy_id: str
    strategy_family: str
    setup_origin_timestamp: str
    signal_timestamp: str
    timeframe_contract: StrategyTimeframeContract
    lifecycle_state: SwingLifecycleState
    entry_price_reference: float
    stop: float
    target: float
    expected_holding_period: str
    expected_net_return: float | None
    expected_net_r: float | None
    current_regime: str
    multi_timeframe_alignment: str
    multi_timeframe_conflict: str | None
    liquidity: float | None
    spread_bps: float | None
    estimated_round_trip_cost: float | None
    whole_share_feasibility: str
    portfolio_fit: str
    shariah_status: str
    authority_status: str

    def __post_init__(self) -> None:
        for name in ("symbol", "security_id", "strategy_id", "strategy_family"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        signal = _utc_timestamp(self.signal_timestamp, "signal_timestamp")
        origin = _utc_timestamp(
            self.setup_origin_timestamp,
            "setup_origin_timestamp",
        )
        if signal < origin:
            raise ValueError("signal_timestamp cannot precede setup_origin_timestamp")
        if self.entry_price_reference <= 0:
            raise ValueError("entry_price_reference must be positive")
        if not 0 < self.stop < self.entry_price_reference:
            raise ValueError("long swing stop must be below entry_price_reference")
        if self.target <= self.entry_price_reference:
            raise ValueError("long swing target must be above entry_price_reference")

    @property
    def setup_id(self) -> str:
        return stable_setup_identity(
            symbol=self.symbol,
            strategy_id=self.strategy_id,
            setup_origin_timestamp=self.setup_origin_timestamp,
            setup_timeframe=self.timeframe_contract.setup_timeframe,
        )

    @property
    def risk_per_share(self) -> float:
        return round(self.entry_price_reference - self.stop, 10)

    def as_dict(self, *, observed_at: datetime | None = None) -> dict[str, Any]:
        observed = pd.Timestamp(observed_at or datetime.now(UTC))
        observed = (
            observed.tz_localize("UTC")
            if observed.tzinfo is None
            else observed.tz_convert("UTC")
        )
        signal = _utc_timestamp(self.signal_timestamp, "signal_timestamp")
        return {
            "schema": "active_swing_opportunity_v1",
            "setup_id": self.setup_id,
            "risk_per_share": self.risk_per_share,
            "signal_age_seconds": max(0.0, (observed - signal).total_seconds()),
            "entry_timeframe": self.timeframe_contract.entry_timeframe,
            "setup_timeframe": self.timeframe_contract.setup_timeframe,
            "context_timeframes": list(self.timeframe_contract.context_timeframes),
            "structural_timeframe": self.timeframe_contract.structural_timeframe,
            **{
                key: (value.value if isinstance(value, StrEnum) else value)
                for key, value in asdict(self).items()
                if key != "timeframe_contract"
            },
            "timeframe_contract": self.timeframe_contract.as_dict(),
            "execution_authority": "NONE",
            "submits_orders": False,
        }


def stable_setup_identity(
    *,
    symbol: str,
    strategy_id: str,
    setup_origin_timestamp: str,
    setup_timeframe: str,
) -> str:
    origin = _utc_timestamp(
        setup_origin_timestamp,
        "setup_origin_timestamp",
    )
    return stable_hash(
        {
            "symbol": str(symbol).strip().upper(),
            "strategy_id": str(strategy_id).strip(),
            "setup_origin_timestamp": origin.isoformat(),
            "setup_timeframe": canonical_interval(setup_timeframe),
        }
    )


def causal_timeframe_asof_join(
    decisions: pd.DataFrame,
    timeframe_states: Mapping[str, pd.DataFrame],
    contract: StrategyTimeframeContract,
) -> pd.DataFrame:
    """Attach the latest fully available state per security and timeframe.

    Every consumed state satisfies bar_close_time <= available_at <=
    decision_timestamp. Missing state is represented with has_<timeframe> and
    never filled with invented values.
    """
    required_decision_columns = {"security_id", "decision_timestamp"}
    missing = required_decision_columns - set(decisions.columns)
    if missing:
        raise ValueError(f"decision columns missing: {sorted(missing)}")
    output = decisions.copy().reset_index(drop=True)
    output["decision_timestamp"] = pd.to_datetime(
        output["decision_timestamp"], utc=True, errors="raise"
    )
    output["_decision_order"] = range(len(output))
    for timeframe in contract.all_timeframes:
        state = timeframe_states.get(timeframe)
        output = _join_one_timeframe(output, state, timeframe)
    required_masks = [f"has_{_column_timeframe(value)}" for value in contract.required_timeframes]
    output["timeframe_contract_valid"] = output[required_masks].all(axis=1)
    output["timeframe_blockers"] = output.apply(
        lambda row: tuple(
            f"REQUIRED_TIMEFRAME_MISSING:{timeframe}"
            for timeframe in contract.required_timeframes
            if not bool(row[f"has_{_column_timeframe(timeframe)}"])
        ),
        axis=1,
    )
    return (
        output.sort_values("_decision_order")
        .drop(columns="_decision_order")
        .reset_index(drop=True)
    )


def resolve_signal_swing_contract(
    signals: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
) -> dict[str, Any]:
    """Resolve only explicit native contracts; never invent timeframe roles."""
    contracts = [
        row.get("strategy_timeframe_contract") or row.get("timeframe_contract")
        for row in signals
        if row.get("strategy_timeframe_contract") or row.get("timeframe_contract")
    ]
    if not contracts:
        return _unresolved_signal_contract(
            "UNDECLARED_RESEARCH_ONLY",
            "STRATEGY_TIMEFRAME_CONTRACT_REQUIRED",
        )
    try:
        resolved = [_contract_from_mapping(value) for value in contracts]
    except (TypeError, ValueError) as exc:
        return _unresolved_signal_contract(
            "INVALID_FAIL_CLOSED",
            f"INVALID_STRATEGY_TIMEFRAME_CONTRACT:{type(exc).__name__}",
        )
    serialized = [item.as_dict() for item in resolved]
    if any(value != serialized[0] for value in serialized[1:]):
        return _unresolved_signal_contract(
            "CONFLICTING_FAIL_CLOSED",
            "CONFLICTING_STRATEGY_TIMEFRAME_CONTRACTS",
        )
    contract = resolved[0]
    observed = {
        canonical_interval(str(row.get("timeframe")))
        for row in signals
        if row.get("timeframe")
    }
    missing = [
        value for value in contract.required_timeframes if value not in observed
    ]
    origins = sorted(
        {
            str(row.get("setup_origin_timestamp"))
            for row in signals
            if row.get("setup_origin_timestamp")
        }
    )
    strategy_ids = sorted(
        {str(row.get("strategy_id")) for row in signals if row.get("strategy_id")}
    )
    blockers = [
        f"REQUIRED_TIMEFRAME_MISSING:{value}" for value in missing
    ]
    if len(origins) != 1:
        blockers.append("STABLE_SETUP_ORIGIN_REQUIRED")
    if len(strategy_ids) != 1:
        blockers.append("SINGLE_ECONOMIC_STRATEGY_ID_REQUIRED")
    setup_id = (
        stable_setup_identity(
            symbol=symbol,
            strategy_id=strategy_ids[0],
            setup_origin_timestamp=origins[0],
            setup_timeframe=contract.setup_timeframe,
        )
        if not blockers
        else None
    )
    explicit_states = {
        str(row.get("lifecycle_state"))
        for row in signals
        if row.get("lifecycle_state")
    }
    if len(explicit_states) > 1:
        blockers.append("CONFLICTING_LIFECYCLE_STATES")
    try:
        lifecycle = (
            SwingLifecycleState(next(iter(explicit_states)))
            if len(explicit_states) == 1
            else SwingLifecycleState.WATCHING
        )
    except ValueError:
        blockers.append("INVALID_LIFECYCLE_STATE")
        lifecycle = SwingLifecycleState.DISCOVERED
    valid = not blockers
    return {
        "schema": "resolved_signal_active_swing_contract_v1",
        "status": "EXPLICIT_VALID" if valid else "EXPLICIT_BLOCKED",
        "lifecycle_state": lifecycle.value,
        "setup_id": setup_id,
        "contract": contract.as_dict(),
        "observed_timeframes": sorted(observed, key=_timeframe_order),
        "blockers": blockers,
        "execution_authority": "NONE",
    }


def _join_one_timeframe(
    decisions: pd.DataFrame,
    state: pd.DataFrame | None,
    timeframe: str,
) -> pd.DataFrame:
    suffix = _column_timeframe(timeframe)
    has_column = f"has_{suffix}"
    if state is None or state.empty:
        result = decisions.copy()
        result[has_column] = False
        return result
    required = {"security_id", "bar_close_time", "available_at"}
    missing = required - set(state.columns)
    if missing:
        raise ValueError(f"{timeframe} state columns missing: {sorted(missing)}")
    right = state.copy()
    right["bar_close_time"] = pd.to_datetime(
        right["bar_close_time"], utc=True, errors="raise"
    )
    right["available_at"] = pd.to_datetime(
        right["available_at"], utc=True, errors="raise"
    )
    if right["bar_close_time"].gt(right["available_at"]).any():
        raise ValueError(f"CAUSALITY_VIOLATION:{timeframe}:bar_close_after_available_at")
    if right.duplicated(["security_id", "available_at"]).any():
        raise ValueError(f"DUPLICATE_TIMEFRAME_STATE:{timeframe}")
    rename = {
        column: f"{column}_{suffix}"
        for column in right.columns
        if column != "security_id"
    }
    right = right.rename(columns=rename)
    available_column = f"available_at_{suffix}"
    result_parts: list[pd.DataFrame] = []
    for security_id, left_group in decisions.groupby("security_id", sort=False):
        right_group = right.loc[right["security_id"].eq(security_id)]
        left_sorted = left_group.sort_values("decision_timestamp")
        if right_group.empty:
            merged = left_sorted.copy()
            for column in rename.values():
                merged[column] = pd.NA
        else:
            merged = pd.merge_asof(
                left_sorted,
                right_group.sort_values(available_column).drop(columns="security_id"),
                left_on="decision_timestamp",
                right_on=available_column,
                direction="backward",
                allow_exact_matches=True,
            )
        result_parts.append(merged)
    result = pd.concat(result_parts, ignore_index=True)
    result[has_column] = result[available_column].notna()
    return result


def _contract_from_mapping(value: Any) -> StrategyTimeframeContract:
    if not isinstance(value, Mapping):
        raise TypeError("timeframe contract must be a mapping")
    return StrategyTimeframeContract(
        entry_timeframe=str(value["entry_timeframe"]),
        setup_timeframe=str(value["setup_timeframe"]),
        context_timeframes=tuple(value.get("context_timeframes", ())),
        structural_timeframe=str(value["structural_timeframe"]),
        management_timeframe=str(value["management_timeframe"]),
        exit_timeframe=str(value["exit_timeframe"]),
        required_timeframes=tuple(value["required_timeframes"]),
        optional_timeframes=tuple(value.get("optional_timeframes", ())),
        session=str(value.get("session", "RTH")),
    )


def _unresolved_signal_contract(status: str, blocker: str) -> dict[str, Any]:
    return {
        "schema": "resolved_signal_active_swing_contract_v1",
        "status": status,
        "lifecycle_state": SwingLifecycleState.DISCOVERED.value,
        "setup_id": None,
        "contract": None,
        "observed_timeframes": [],
        "blockers": [blocker],
        "execution_authority": "NONE",
    }


def _canonical_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {canonical_interval(value) for value in values},
            key=_timeframe_order,
        )
    )


def _timeframe_order(value: str) -> int:
    try:
        return CANONICAL_INTERVALS.index(canonical_interval(value))
    except ValueError:
        return len(CANONICAL_INTERVALS)


def _column_timeframe(value: str) -> str:
    return canonical_interval(value).replace("mo", "month")


def _utc_timestamp(value: Any, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a timestamp") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a timestamp")
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


__all__ = [
    "ActiveSwingOpportunity",
    "StrategyTimeframeContract",
    "SwingLifecycleState",
    "causal_timeframe_asof_join",
    "resolve_signal_swing_contract",
    "stable_setup_identity",
]
