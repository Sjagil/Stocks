from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any, Iterable

from stocks.research.autopilot.components import (
    MACRO_COMPONENT_NAMES,
    component_registry,
)
from stocks.research.autopilot.contracts import (
    ResearchBudgets,
    StrategyFamily,
    StrategySpec,
    StrategyStatus,
    canonical_swing_timeframe,
    stable_hash,
)


HYPOTHESES = {
    StrategyFamily.QUALITY_MOMENTUM: (
        "Shariah-eligible profitable companies with positive six- and twelve-month "
        "momentum and a positive long-term trend tend to persist."
    ),
    StrategyFamily.TREND_PULLBACK: (
        "A bounded pullback toward a medium EMA inside confirmed daily and weekly "
        "uptrends offers a better swing entry than chasing an extended move."
    ),
    StrategyFamily.ETF_ROTATION: (
        "Liquid ordinary ETFs with superior medium-term relative strength and a "
        "positive trend tend to retain leadership between scheduled rebalances."
    ),
    StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT: (
        "Volatility contraction inside a positive trend followed by a confirmed "
        "price and volume breakout can precede directional expansion."
    ),
    StrategyFamily.COMMODITY_ETF_TREND: (
        "Approved unlevered commodity products with aligned daily and weekly trends "
        "can provide persistent long-only exposure while cash absorbs negative regimes."
    ),
}

FAMILY_PARAMETER_BOUNDS: dict[
    StrategyFamily, dict[str, tuple[float, float] | type]
] = {
    StrategyFamily.QUALITY_MOMENTUM: {
        "top_n": (5, 25),
        "ema_period": (100, 300),
        "momentum_6m": (63, 189),
        "momentum_12m": (189, 378),
    },
    StrategyFamily.TREND_PULLBACK: {
        "pullback_ema": (10, 100),
        "tolerance": (0.005, 0.10),
        "atr_multiple": (1.0, 6.0),
        "max_hold": (5, 126),
    },
    StrategyFamily.ETF_ROTATION: {
        "top_n": (1, 15),
        "trend_period": (100, 300),
        "cash_when_negative": bool,
    },
    StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT: {
        "breakout_period": (10, 100),
        "vol_short": (5, 30),
        "vol_long": (20, 150),
        "contraction": (0.30, 0.90),
        "atr_multiple": (1.0, 6.0),
    },
    StrategyFamily.COMMODITY_ETF_TREND: {
        "top_n": (1, 5),
        "fast": (10, 150),
        "slow": (100, 400),
        "cash_when_negative": bool,
    },
}


def generate_strategies(
    *,
    budget: int,
    family: str | None = None,
    seed: int = 20260726,
    budgets: ResearchBudgets = ResearchBudgets(),
) -> list[StrategySpec]:
    budgets.validate()
    if budget <= 0 or budget > budgets.max_new_strategies_per_day:
        raise ValueError(
            f"budget must be in [1,{budgets.max_new_strategies_per_day}]"
        )
    families = (
        [StrategyFamily(family)]
        if family is not None
        else list(StrategyFamily)
    )
    candidates: list[StrategySpec] = []
    for selected_family in families:
        variants = list(_family_variants(selected_family, seed=seed))
        if len(variants) > budgets.max_trials_per_family:
            variants = variants[: budgets.max_trials_per_family]
        candidates.extend(variants)
    candidates = sorted(candidates, key=lambda item: item.strategy_id)
    unique: dict[str, StrategySpec] = {}
    for candidate in candidates:
        if candidate.strategy_hash in unique:
            continue
        validate_strategy(candidate)
        unique[candidate.strategy_hash] = candidate
        if len(unique) >= budget:
            break
    return list(unique.values())


def validate_strategy(strategy: StrategySpec) -> None:
    registry = component_registry()
    canonical_swing_timeframe(strategy.entry_timeframe)
    if strategy.confirmation_timeframe:
        canonical_swing_timeframe(strategy.confirmation_timeframe)
    if strategy.regime_timeframe:
        canonical_swing_timeframe(strategy.regime_timeframe)
    if len(strategy.entry_components) > 3:
        raise ValueError("TOO_MANY_ENTRY_CONDITIONS")
    if len(strategy.confirmation_components) > 3:
        raise ValueError("TOO_MANY_CONFIRMATION_FILTERS")
    if len(strategy.regime_components) > 2:
        raise ValueError("TOO_MANY_REGIME_FILTERS")
    if len(strategy.exit_components) > 3:
        raise ValueError("TOO_MANY_EXIT_RULES")
    names = (
        strategy.entry_components
        + strategy.confirmation_components
        + strategy.regime_components
        + strategy.exit_components
        + (strategy.sizing_component,)
    )
    missing = sorted(set(names) - set(registry))
    if missing:
        raise ValueError(f"UNREGISTERED_COMPONENTS:{','.join(missing)}")
    for name in strategy.entry_components:
        if strategy.entry_timeframe not in registry[name].supported_timeframes:
            raise ValueError(f"INCOMPATIBLE_TIMEFRAME:{name}:{strategy.entry_timeframe}")
    if strategy.confirmation_timeframe:
        for name in strategy.confirmation_components:
            if strategy.confirmation_timeframe not in registry[name].supported_timeframes:
                raise ValueError(
                    f"INCOMPATIBLE_TIMEFRAME:{name}:{strategy.confirmation_timeframe}"
                )
    if strategy.regime_timeframe:
        for name in strategy.regime_components:
            if strategy.regime_timeframe not in registry[name].supported_timeframes:
                raise ValueError(
                    f"INCOMPATIBLE_TIMEFRAME:{name}:{strategy.regime_timeframe}"
                )
    if StrategyFamily(strategy.family) == StrategyFamily.TREND_PULLBACK and (
        "benchmark_trend" not in strategy.regime_components
    ):
        raise ValueError("MEAN_REVERSION_REQUIRES_POSITIVE_TREND_REGIME")
    if strategy.portfolio_model not in {
        "equal_weight",
        "inverse_volatility",
        "score_weight",
        "rank_weight",
        "capped_risk_adjusted",
        "sector_first",
        "regional_sleeves",
        "etf_rotation",
    }:
        raise ValueError("UNSUPPORTED_PORTFOLIO_MODEL")
    macro_filters = set(strategy.regime_components) & MACRO_COMPONENT_NAMES
    if len(macro_filters) > 2:
        raise ValueError("MAXIMUM_TWO_MACRO_FILTERS")
    if macro_filters and not strategy.entry_components:
        raise ValueError("MACRO_ONLY_STRATEGY_BLOCKED")
    forbidden_assets = {
        "FUT",
        "OPTION",
        "CFD",
        "CRYPTO",
        "BOND",
        "INVERSE_ETF",
        "LEVERAGED_ETF",
    }
    if set(strategy.asset_scope) & forbidden_assets:
        raise ValueError("FORBIDDEN_ASSET_SCOPE")
    if not strategy.long_only or strategy.leverage_allowed or strategy.shorting_allowed:
        raise ValueError("LONG_ONLY_AUTHORITY_CONTRACT_VIOLATION")
    _validate_parameters(strategy)
    if strategy.seed < 0:
        raise ValueError("seed must be non-negative")
    expected = stable_hash(strategy.core_payload())
    if strategy.strategy_hash != expected:
        raise ValueError("STRATEGY_HASH_MISMATCH")


def generate_macro_variant(
    strategy: StrategySpec,
    macro_filters: tuple[str, ...],
) -> StrategySpec:
    if not 1 <= len(macro_filters) <= 2:
        raise ValueError("MACRO_FILTER_COUNT_MUST_BE_ONE_OR_TWO")
    if set(macro_filters) - MACRO_COMPONENT_NAMES:
        raise ValueError("UNKNOWN_MACRO_FILTER")
    non_macro = tuple(
        component
        for component in strategy.regime_components
        if component not in MACRO_COMPONENT_NAMES
    )
    retained_non_macro = non_macro[: max(0, 2 - len(macro_filters))]
    payload = strategy.core_payload()
    payload.update(
        {
            "regime_components": tuple(
                dict.fromkeys((*retained_non_macro, *macro_filters))
            ),
            "hypothesis": (
                f"{strategy.hypothesis} Macrofilters beperken blootstelling "
                "alleen wanneer point-in-time context dit bevestigt."
            ),
            "parent_strategy_id": strategy.strategy_id,
            "mutation_type": "MACRO_FILTER_ADDITION",
            "status": StrategyStatus.GENERATED,
        }
    )
    payload.pop("status", None)
    digest = stable_hash(payload)
    variant = StrategySpec(
        strategy_id=f"{strategy.family.upper()}-MACRO-{digest[:16]}",
        strategy_hash=digest,
        status=StrategyStatus.GENERATED,
        **payload,
    )
    validate_strategy(variant)
    return variant


def _validate_parameters(strategy: StrategySpec) -> None:
    family = StrategyFamily(strategy.family)
    bounds = FAMILY_PARAMETER_BOUNDS[family]
    unknown = set(strategy.parameters) - set(bounds)
    missing = set(bounds) - set(strategy.parameters)
    if unknown:
        raise ValueError(f"UNREGISTERED_PARAMETERS:{','.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"MISSING_PARAMETERS:{','.join(sorted(missing))}")
    for name, contract in bounds.items():
        value = strategy.parameters[name]
        if contract is bool:
            if not isinstance(value, bool):
                raise ValueError(f"INVALID_PARAMETER_TYPE:{name}")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"INVALID_PARAMETER_TYPE:{name}")
        assert isinstance(contract, tuple)
        if not contract[0] <= float(value) <= contract[1]:
            raise ValueError(f"PARAMETER_OUT_OF_BOUNDS:{name}")
    if family == StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT and (
        float(strategy.parameters["vol_short"])
        >= float(strategy.parameters["vol_long"])
    ):
        raise ValueError("PARAMETER_RELATION_VIOLATION:vol_short<vol_long")
    if family == StrategyFamily.COMMODITY_ETF_TREND and (
        float(strategy.parameters["fast"]) >= float(strategy.parameters["slow"])
    ):
        raise ValueError("PARAMETER_RELATION_VIOLATION:fast<slow")


def near_duplicate_fingerprint(strategy: StrategySpec) -> str:
    payload = {
        "family": strategy.family,
        "entry_timeframe": strategy.entry_timeframe,
        "confirmation_timeframe": strategy.confirmation_timeframe,
        "regime_timeframe": strategy.regime_timeframe,
        "entry_components": strategy.entry_components,
        "confirmation_components": strategy.confirmation_components,
        "regime_components": strategy.regime_components,
        "exit_components": strategy.exit_components,
        "sizing_component": strategy.sizing_component,
        "asset_scope": strategy.asset_scope,
        "long_only": strategy.long_only,
        "portfolio_model": strategy.portfolio_model,
        "rebalance": strategy.rebalance,
    }
    return stable_hash(payload)


def _family_variants(
    family: StrategyFamily,
    *,
    seed: int,
) -> Iterable[StrategySpec]:
    if family == StrategyFamily.QUALITY_MOMENTUM:
        for top_n, model in itertools.product((10, 15), ("equal_weight", "inverse_volatility")):
            yield _build(
                family,
                seed,
                entry_timeframe="1d",
                confirmation_timeframe="1w",
                regime_timeframe=None,
                entries=("quality_composite", "momentum_6m", "momentum_12m"),
                confirmations=("price_above_ma",),
                regimes=("benchmark_trend",),
                exits=("ranking_exit", "fundamental_eligibility_exit", "rebalance_exit"),
                sizing="equal_weight_sizing" if model == "equal_weight" else "inverse_volatility_sizing",
                parameters={"top_n": top_n, "ema_period": 200, "momentum_6m": 126, "momentum_12m": 252},
                portfolio_model=model,
                rebalance="MONTHLY",
            )
    elif family == StrategyFamily.TREND_PULLBACK:
        for timeframe, ema, exit_name in itertools.product(
            ("1h", "4h"),
            (20, 50),
            ("atr_trailing_exit", "moving_average_exit"),
        ):
            yield _build(
                family,
                seed,
                entry_timeframe=timeframe,
                confirmation_timeframe="1d",
                regime_timeframe="1w",
                entries=("ema_pullback",),
                confirmations=("ma_alignment", "relative_volume", "momentum_6m"),
                regimes=("benchmark_trend",),
                exits=(exit_name, "time_stop"),
                sizing="volatility_adjusted_sizing",
                parameters={"pullback_ema": ema, "tolerance": 0.05, "atr_multiple": 3.0, "max_hold": 30},
                portfolio_model="capped_risk_adjusted",
                rebalance="DAILY_AFTER_CLOSE",
            )
    elif family == StrategyFamily.ETF_ROTATION:
        for top_n, rebalance in itertools.product((3, 5, 10), ("MONTHLY", "QUARTERLY")):
            yield _build(
                family,
                seed,
                entry_timeframe="1d",
                confirmation_timeframe="1w",
                regime_timeframe=None,
                entries=("momentum_3m", "momentum_6m", "momentum_12m"),
                confirmations=("relative_strength", "price_above_ma"),
                regimes=("benchmark_trend",),
                exits=("ranking_exit", "rebalance_exit", "regime_exit"),
                sizing="equal_weight_sizing",
                parameters={"top_n": top_n, "trend_period": 200, "cash_when_negative": True},
                portfolio_model="equal_weight",
                rebalance=rebalance,
            )
    elif family == StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT:
        for timeframe, breakout in itertools.product(("4h", "1d"), (20, 55)):
            yield _build(
                family,
                seed,
                entry_timeframe=timeframe,
                confirmation_timeframe="1d" if timeframe == "4h" else "1w",
                regime_timeframe=None,
                entries=("volatility_contraction", "donchian_breakout"),
                confirmations=("volume_confirmation", "ma_alignment"),
                regimes=("benchmark_trend",),
                exits=("atr_trailing_exit", "time_stop"),
                sizing="volatility_adjusted_sizing",
                parameters={"breakout_period": breakout, "vol_short": 10, "vol_long": 60, "contraction": 0.65, "atr_multiple": 3.0},
                portfolio_model="capped_risk_adjusted",
                rebalance="DAILY_AFTER_CLOSE",
            )
    elif family == StrategyFamily.COMMODITY_ETF_TREND:
        for timeframe, top_n in itertools.product(("1d", "1w"), (1, 2)):
            yield _build(
                family,
                seed,
                entry_timeframe=timeframe,
                confirmation_timeframe="1w" if timeframe == "1d" else "1mo",
                regime_timeframe=None,
                entries=("ma_alignment", "momentum_6m"),
                confirmations=("relative_strength",),
                regimes=("commodity_regime",),
                exits=("moving_average_exit", "regime_exit", "rebalance_exit"),
                sizing="inverse_volatility_sizing",
                parameters={"top_n": top_n, "fast": 50, "slow": 200, "cash_when_negative": True},
                portfolio_model="inverse_volatility",
                rebalance="MONTHLY",
            )


def _build(
    family: StrategyFamily,
    seed: int,
    *,
    entry_timeframe: str,
    confirmation_timeframe: str | None,
    regime_timeframe: str | None,
    entries: tuple[str, ...],
    confirmations: tuple[str, ...],
    regimes: tuple[str, ...],
    exits: tuple[str, ...],
    sizing: str,
    parameters: dict[str, Any],
    portfolio_model: str,
    rebalance: str,
) -> StrategySpec:
    core: dict[str, Any] = {
        "version": "1.1.0",
        "family": family.value,
        "hypothesis": HYPOTHESES[family],
        "entry_timeframe": entry_timeframe,
        "confirmation_timeframe": confirmation_timeframe,
        "regime_timeframe": regime_timeframe,
        "entry_components": entries,
        "confirmation_components": confirmations,
        "regime_components": regimes,
        "exit_components": exits,
        "sizing_component": sizing,
        "asset_scope": _asset_scope(family),
        "long_only": True,
        "leverage_allowed": False,
        "shorting_allowed": False,
        "parameters": parameters,
        "portfolio_model": portfolio_model,
        "rebalance": rebalance,
        "seed": seed,
        "parent_strategy_id": None,
        "mutation_type": "EXACT_TEMPLATE",
    }
    digest = stable_hash(core)
    strategy = StrategySpec(
        strategy_id=f"{family.value.upper()}-{digest[:16]}",
        strategy_hash=digest,
        **core,
    )
    return replace(strategy)


def _asset_scope(family: StrategyFamily) -> tuple[str, ...]:
    if family in {StrategyFamily.QUALITY_MOMENTUM, StrategyFamily.TREND_PULLBACK}:
        return ("STOCK",)
    if family == StrategyFamily.ETF_ROTATION:
        return ("ETF", "SECTOR_ETF", "REGIONAL_ETF", "SHARIAH_ETF")
    if family == StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT:
        return ("STOCK", "ETF")
    return ("COMMODITY_ETF", "COMMODITY_ETC", "ETC")
