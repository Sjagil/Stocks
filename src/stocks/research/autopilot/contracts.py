from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import pandas as pd


ALLOWED_SWING_TIMEFRAMES = (
    "15m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
    "1w",
    "1mo",
)
FORBIDDEN_TIMEFRAMES = frozenset({"1m", "5m", "30m", "tick", "ticks"})
TIMEFRAME_ALIASES = {
    "15min": "15m",
    "1month": "1mo",
    "month": "1mo",
    "1M": "1mo",
    "1wk": "1w",
    "60m": "1h",
}
PRIMARY_TIMEFRAME_COMBINATIONS = (
    ("15m", "1h", "4h"),
    ("15m", "4h", "1d"),
    ("1h", "4h", "1d"),
    ("4h", "1d", "1w"),
    ("1d", "1w", None),
    ("1w", "1mo", None),
)


class StrategyStatus(StrEnum):
    GENERATED = "GENERATED"
    EXPERIMENTAL = "EXPERIMENTAL"
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
    FROZEN_SHADOW = "FROZEN_SHADOW"
    MANUAL_SIGNAL_CANDIDATE = "MANUAL_SIGNAL_CANDIDATE"
    LIVE_CANARY_CANDIDATE = "LIVE_CANARY_CANDIDATE"
    CONTROLLED_LIVE = "CONTROLLED_LIVE"
    SMOKE_PASS = "SMOKE_PASS"
    RESEARCH_PASS = "RESEARCH_PASS"
    ROBUSTNESS_PASS = "ROBUSTNESS_PASS"
    PORTFOLIO_PASS = "PORTFOLIO_PASS"
    FORWARD_OBSERVER = "FORWARD_OBSERVER"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


class ResearchLevel(StrEnum):
    EXPERIMENTAL = "EXPERIMENTAL"
    RESEARCH_WATCHLIST = "RESEARCH_WATCHLIST"
    FROZEN_SHADOW = "FROZEN_SHADOW"
    MANUAL_SIGNAL_CANDIDATE = "MANUAL_SIGNAL_CANDIDATE"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    LIVE_CANARY_CANDIDATE = "LIVE_CANARY_CANDIDATE"
    CONTROLLED_LIVE = "CONTROLLED_LIVE"
    FORWARD_OBSERVER_CANDIDATE = "FORWARD_OBSERVER_CANDIDATE"
    FINANCIAL_FINALIST = "FINANCIAL_FINALIST"
    NO_CLASSIFICATION = "NO_CLASSIFICATION"


class StrategyFamily(StrEnum):
    QUALITY_MOMENTUM = "quality_momentum"
    TREND_PULLBACK = "trend_pullback"
    ETF_ROTATION = "etf_rotation"
    VOLATILITY_CONTRACTION_BREAKOUT = "volatility_contraction_breakout"
    COMMODITY_ETF_TREND = "commodity_etf_trend"


@dataclass(frozen=True)
class ResearchBudgets:
    max_new_strategies_per_day: int = 100
    max_trials_per_family: int = 40
    max_parameter_combinations: int = 250
    max_runtime_seconds: int = 1_800
    max_storage_mb: int = 2_000
    max_variants_per_parent: int = 12

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    version: str
    category: str
    formula: str
    required_columns: tuple[str, ...]
    supported_timeframes: tuple[str, ...]
    minimum_history: int
    parameter_bounds: dict[str, tuple[float, float]]
    lookback: int
    warmup: int
    causality_rule: str
    asset_compatibility: tuple[str, ...]
    output_type: str
    missing_data_policy: str
    test_status: str
    unit: str
    output_range: str
    causality_status: str

    def validate(self) -> None:
        if not self.name or not self.version or not self.formula:
            raise ValueError("component identity and formula are required")
        if self.minimum_history < 0 or self.lookback < 0 or self.warmup < 0:
            raise ValueError(f"negative history contract for {self.name}")
        for timeframe in self.supported_timeframes:
            canonical_swing_timeframe(timeframe)
        for parameter, bounds in self.parameter_bounds.items():
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"invalid bounds for {self.name}.{parameter}")


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    strategy_hash: str
    version: str
    family: str
    hypothesis: str
    entry_timeframe: str
    confirmation_timeframe: str | None
    regime_timeframe: str | None
    entry_components: tuple[str, ...]
    confirmation_components: tuple[str, ...]
    regime_components: tuple[str, ...]
    exit_components: tuple[str, ...]
    sizing_component: str
    asset_scope: tuple[str, ...]
    long_only: bool
    leverage_allowed: bool
    shorting_allowed: bool
    parameters: dict[str, Any]
    portfolio_model: str
    rebalance: str
    seed: int
    parent_strategy_id: str | None
    mutation_type: str
    status: str = StrategyStatus.GENERATED

    def core_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("strategy_id")
        payload.pop("strategy_hash")
        payload.pop("status")
        return payload


def canonical_swing_timeframe(value: str) -> str:
    raw = str(value).strip()
    normalized = TIMEFRAME_ALIASES.get(raw, raw.lower())
    if normalized in FORBIDDEN_TIMEFRAMES:
        raise ValueError(f"FORBIDDEN_SWING_TIMEFRAME:{normalized}")
    if normalized not in ALLOWED_SWING_TIMEFRAMES:
        raise ValueError(f"UNSUPPORTED_SWING_TIMEFRAME:{value}")
    return normalized


def stable_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(canonical).hexdigest().upper()


def validate_closed_candles(
    frame: pd.DataFrame,
    *,
    decision_time: datetime | pd.Timestamp,
) -> pd.DataFrame:
    if "bar_close_utc" not in frame:
        raise ValueError("bar_close_utc is required")
    closes = pd.to_datetime(frame["bar_close_utc"], utc=True, errors="coerce")
    if closes.isna().any():
        raise ValueError("invalid bar_close_utc")
    decision = pd.Timestamp(decision_time)
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    else:
        decision = decision.tz_convert("UTC")
    return frame.loc[closes <= decision].copy()


def causal_higher_timeframe_map(
    lower: pd.DataFrame,
    higher: pd.DataFrame,
    *,
    lower_decision_column: str = "bar_close_utc",
    higher_available_column: str = "bar_close_utc",
) -> pd.DataFrame:
    if lower_decision_column not in lower or higher_available_column not in higher:
        raise ValueError("causal mapping timestamp columns are required")
    left = lower.copy()
    right = higher.copy()
    left["_decision_utc"] = pd.to_datetime(left[lower_decision_column], utc=True, errors="raise")
    right["_available_utc"] = pd.to_datetime(
        right[higher_available_column], utc=True, errors="raise"
    )
    left = left.sort_values("_decision_utc")
    right = right.sort_values("_available_utc")
    feature_columns = [
        column
        for column in right.columns
        if column not in {higher_available_column, "_available_utc"}
    ]
    renamed = right[["_available_utc", *feature_columns]].rename(
        columns={column: f"htf_{column}" for column in feature_columns}
    )
    mapped = pd.merge_asof(
        left,
        renamed,
        left_on="_decision_utc",
        right_on="_available_utc",
        direction="backward",
        allow_exact_matches=True,
    )
    future = mapped["_available_utc"].notna() & (mapped["_available_utc"] > mapped["_decision_utc"])
    if bool(future.any()):
        raise AssertionError("higher-timeframe lookahead detected")
    return mapped.drop(columns=["_decision_utc", "_available_utc"])
