from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class ShadowDecisionRequest:
    strategy_id: str
    strategy_version: str
    decision_timestamp: str
    information_cutoff_timestamp: str
    first_executable_timestamp: str
    dataset_manifest_hash: str
    dataset_content_hashes: dict[str, str]
    universe_hash: str
    parameter_hash: str


@dataclass(frozen=True)
class ShadowSignal:
    decision_id: str
    con_id: int
    feature_name: str
    feature_value: Decimal
    feature_timestamp: str
    available_at: str
    source_dataset: str
    source_content_hash: str
    calculation_version: str
    signal_value: Decimal
    signal_status: str


@dataclass(frozen=True)
class ShadowTargetPosition:
    con_id: int
    symbol: str
    region: str
    sleeve: str
    currency: str
    target_weight: Decimal


@dataclass(frozen=True)
class ShadowTargetPortfolio:
    decision_id: str
    positions: tuple[ShadowTargetPosition, ...]
    cash_weight: Decimal
    target_portfolio_hash: str
    status: str


@dataclass(frozen=True)
class ShadowDecision:
    decision_id: str
    strategy_id: str
    strategy_version: str
    strategy_hash: str
    decision_timestamp: str
    information_cutoff_timestamp: str
    first_executable_timestamp: str
    dataset_manifest_hash: str
    dataset_content_hashes: dict[str, str]
    universe_hash: str
    parameter_hash: str
    eligible_instruments: tuple[int, ...]
    blocked_instruments: tuple[int, ...]
    block_reasons: dict[int, str]
    signal_count: int
    target_portfolio_hash: str
    authority: str
    status: str
    created_at: str


@dataclass(frozen=True)
class ShadowFill:
    fill_id: str
    decision_id: str
    con_id: int
    quantity: Decimal
    price: Decimal
    execution_proxy: str
    price_source: str
    price_timestamp: str
    slippage_bps: Decimal
    commission_bps: Decimal
    spread_bps: Decimal
    fill_status: str


@dataclass(frozen=True)
class ShadowPosition:
    con_id: int
    quantity: Decimal
    market_value: Decimal
    weight: Decimal


@dataclass(frozen=True)
class ShadowPortfolioSnapshot:
    snapshot_id: str
    decision_id: str
    positions: tuple[ShadowPosition, ...]
    cash: Decimal
    nav: Decimal
    fees: Decimal
    created_at: str
    state_hash: str


@dataclass(frozen=True)
class ShadowEvaluation:
    decision_id: str
    decision_timestamp: str
    evaluation_start: str
    evaluation_end: str
    realized_return: Decimal | None
    benchmark_return: Decimal | None
    active_return: Decimal | None
    maximum_adverse_excursion: Decimal | None
    maximum_favorable_excursion: Decimal | None
    costs: Decimal
    evaluation_status: str


@dataclass(frozen=True)
class ShadowBenchmarkComparison:
    decision_id: str
    benchmark_id: str
    shadow_return: Decimal
    benchmark_return: Decimal
    active_return: Decimal
    tracking_error: Decimal
    drawdown_difference: Decimal
    cost_difference: Decimal


@dataclass(frozen=True)
class ShadowDecisionManifest:
    decision_id: str
    decision_hash: str
    signal_hash: str
    target_hash: str
    fill_hash: str
    snapshot_hash: str
    evaluation_hash: str


def model_to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return model_to_jsonable(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return [model_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [model_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): model_to_jsonable(item) for key, item in value.items()}
    return value
