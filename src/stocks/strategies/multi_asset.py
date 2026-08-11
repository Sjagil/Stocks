from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from stocks.data.bars import (
    BarCacheLayout,
    BarDataSource,
    BarDataType,
    BarInterval,
    read_bar_cache_records,
    validate_bar_cache,
)
from stocks.domain.assets import IbkrSecurityType
from stocks.portfolio.sleeves import SleeveName


DEFAULT_SLEEVE_RISK_BUDGETS: dict[SleeveName, float] = {
    SleeveName.EQUITY_MOMENTUM: 0.30,
    SleeveName.ETF_CORE_ROTATION: 0.35,
    SleeveName.BOND_DURATION: 0.10,
    SleeveName.COMMODITY_TREND: 0.15,
    SleeveName.COMMODITY_CARRY: 0.0,
    SleeveName.MEAN_REVERSION: 0.0,
    SleeveName.DEFENSIVE_CASH: 0.10,
}


@dataclass(frozen=True)
class StrategyAssetSeries:
    con_id: int
    security_type: IbkrSecurityType
    sleeve: SleeveName
    timestamps: tuple[datetime, ...]
    closes: tuple[float, ...]

    def validate(self) -> None:
        if self.con_id <= 0:
            raise ValueError("con_id must be positive")
        if len(self.timestamps) != len(self.closes):
            raise ValueError("timestamps and closes must have the same length")
        if len(self.timestamps) < 2:
            raise ValueError("at least two bars are required")
        if tuple(sorted(self.timestamps)) != self.timestamps:
            raise ValueError("timestamps must be sorted ascending")
        if len(set(self.timestamps)) != len(self.timestamps):
            raise ValueError("timestamps must be unique")
        if any(close <= 0 for close in self.closes):
            raise ValueError("closes must be positive")

    def close_by_timestamp(self) -> dict[datetime, float]:
        self.validate()
        return dict(zip(self.timestamps, self.closes, strict=True))


@dataclass(frozen=True)
class MultiAssetBacktestConfig:
    lookback_bars: int = 3
    top_n_per_sleeve: int = 2
    cost_bps: float = 10.0
    initial_nav: float = 100_000.0
    max_asset_weight: float = 0.25
    min_score: float = 0.0
    periods_per_year: float = 252.0
    sleeve_risk_budgets: dict[SleeveName, float] = field(
        default_factory=lambda: dict(DEFAULT_SLEEVE_RISK_BUDGETS)
    )

    def validate(self) -> None:
        if self.lookback_bars < 2:
            raise ValueError("lookback_bars must be at least 2")
        if self.top_n_per_sleeve <= 0:
            raise ValueError("top_n_per_sleeve must be positive")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        if self.initial_nav <= 0:
            raise ValueError("initial_nav must be positive")
        if not 0 < self.max_asset_weight <= 1:
            raise ValueError("max_asset_weight must be in (0, 1]")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        for sleeve, budget in self.sleeve_risk_budgets.items():
            if not isinstance(sleeve, SleeveName):
                raise ValueError("sleeve_risk_budgets keys must be SleeveName values")
            if budget < 0:
                raise ValueError("sleeve budgets cannot be negative")
        risky_budget = sum(
            budget
            for sleeve, budget in self.sleeve_risk_budgets.items()
            if sleeve != SleeveName.DEFENSIVE_CASH
        )
        if risky_budget > 1.0:
            raise ValueError("risky sleeve budgets cannot exceed 1")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "lookback_bars": self.lookback_bars,
            "top_n_per_sleeve": self.top_n_per_sleeve,
            "cost_bps": self.cost_bps,
            "initial_nav": self.initial_nav,
            "max_asset_weight": self.max_asset_weight,
            "min_score": self.min_score,
            "periods_per_year": self.periods_per_year,
            "sleeve_risk_budgets": {
                sleeve.value: budget for sleeve, budget in self.sleeve_risk_budgets.items()
            },
        }


def multi_asset_strategy_schema() -> dict[str, Any]:
    return {
        "schema": "multi_asset_strategy_schema_v1",
        "status": "GO",
        "strategy": "offline_multi_asset_rotation",
        "sleeves": [sleeve.value for sleeve in DEFAULT_SLEEVE_RISK_BUDGETS],
        "signals": [
            "lookback momentum",
            "price above lookback average trend gate",
            "volatility-adjusted score",
            "sleeve risk budgets",
            "defensive cash residual",
        ],
        "metrics": [
            "instrument_count",
            "date_range",
            "bar_count",
            "rebalance_count",
            "gross_return",
            "net_return",
            "CAGR",
            "annualized_volatility",
            "Sharpe",
            "Sortino",
            "total_return",
            "maximum_drawdown",
            "max_drawdown",
            "Calmar",
            "profit_factor",
            "expectancy",
            "win_rate",
            "average_win",
            "average_loss",
            "turnover",
            "total_costs",
            "cash_average",
            "cash_maximum",
            "sleeve_exposure",
            "region_exposure",
        ],
        "execution": {
            "live_orders_enabled": False,
            "provider_calls_enabled": False,
            "uses_local_bar_cache_only": True,
        },
        "financial_calls": _zero_financial_calls(),
    }


def load_strategy_series_from_bar_cache(
    layout: BarCacheLayout,
    *,
    interval: BarInterval,
    data_type: BarDataType,
    source: BarDataSource | None,
) -> tuple[list[StrategyAssetSeries], dict[str, Any]]:
    validation = validate_bar_cache(layout)
    if validation["status"] != "GO":
        return [], validation

    series_by_identity: dict[tuple[int, IbkrSecurityType], list[tuple[datetime, float]]] = {}
    for file_summary in validation["files"]:
        if file_summary["errors"]:
            continue
        if file_summary["partition"]["interval"] != interval.value:
            continue
        if file_summary["partition"]["data_type"] != data_type.value:
            continue
        if source is not None and file_summary["partition"]["source"] != source.value:
            continue
        path = Path(file_summary["path"])
        for record in read_bar_cache_records(path):
            security_type = IbkrSecurityType(record["sec_type"])
            con_id = int(record["con_id"])
            series_by_identity.setdefault((con_id, security_type), []).append(
                (_parse_datetime(record["timestamp_utc"]), float(record["close"]))
            )

    series: list[StrategyAssetSeries] = []
    for (con_id, security_type), rows in sorted(series_by_identity.items()):
        rows = sorted(rows, key=lambda item: item[0])
        sleeve = _default_sleeve_for_security_type(security_type)
        asset_series = StrategyAssetSeries(
            con_id=con_id,
            security_type=security_type,
            sleeve=sleeve,
            timestamps=tuple(timestamp for timestamp, _ in rows),
            closes=tuple(close for _, close in rows),
        )
        asset_series.validate()
        series.append(asset_series)
    return series, validation


def run_multi_asset_rotation_backtest(
    series: list[StrategyAssetSeries],
    *,
    config: MultiAssetBacktestConfig | None = None,
) -> dict[str, Any]:
    effective_config = config or MultiAssetBacktestConfig()
    effective_config.validate()
    if not series:
        return _empty_backtest_report(effective_config, status="NO_DATA", reason="no asset series supplied")

    for asset in series:
        asset.validate()

    all_timestamps = sorted({timestamp for asset in series for timestamp in asset.timestamps})
    if len(all_timestamps) <= effective_config.lookback_bars:
        return _empty_backtest_report(
            effective_config,
            status="NO_DATA",
            reason="not enough bars for configured lookback",
        )

    price_maps = {asset.con_id: asset.close_by_timestamp() for asset in series}
    asset_by_id = {asset.con_id: asset for asset in series}
    nav = effective_config.initial_nav
    peak_nav = nav
    max_drawdown = 0.0
    previous_weights: dict[str, float] = {"CASH": 1.0}
    period_reports: list[dict[str, Any]] = []
    winning_pnl = 0.0
    losing_pnl = 0.0
    total_turnover = 0.0
    trade_count = 0
    total_gross_return = 0.0
    total_cost_return = 0.0
    total_cost_amount = 0.0
    net_returns: list[float] = []
    cash_weights: list[float] = []
    sleeve_exposure_accumulator: dict[str, float] = {}
    region_exposure_accumulator: dict[str, float] = {}

    for decision_index in range(effective_config.lookback_bars, len(all_timestamps) - 1):
        decision_time = all_timestamps[decision_index]
        next_time = all_timestamps[decision_index + 1]
        signals = _build_signals(
            series,
            all_timestamps=all_timestamps,
            decision_time=decision_time,
            config=effective_config,
        )
        target_weights = _target_weights(signals, effective_config)
        period_returns = {
            str(con_id): price_maps[con_id][next_time] / price_maps[con_id][decision_time] - 1.0
            for con_id in asset_by_id
            if decision_time in price_maps[con_id] and next_time in price_maps[con_id]
        }
        turnover = _one_way_turnover(previous_weights, target_weights)
        gross_return = sum(
            weight * period_returns.get(key, 0.0)
            for key, weight in target_weights.items()
            if key != "CASH"
        )
        cost_return = turnover * (effective_config.cost_bps / 10_000.0)
        net_return = gross_return - cost_return
        starting_nav = nav
        cost_amount = starting_nav * cost_return
        pnl = starting_nav * net_return
        nav = starting_nav + pnl
        peak_nav = max(peak_nav, nav)
        drawdown = nav / peak_nav - 1.0
        max_drawdown = min(max_drawdown, drawdown)
        if pnl > 0:
            winning_pnl += pnl
        elif pnl < 0:
            losing_pnl += pnl
        if turnover > 0:
            trade_count += 1
        total_turnover += turnover
        total_gross_return += gross_return
        total_cost_return += cost_return
        total_cost_amount += cost_amount
        net_returns.append(net_return)
        cash_weights.append(target_weights.get("CASH", 0.0))
        _accumulate_exposures(
            sleeve_exposure_accumulator,
            _sleeve_exposure(target_weights, asset_by_id),
        )
        _accumulate_exposures(
            region_exposure_accumulator,
            _region_exposure(target_weights, asset_by_id),
        )
        period_reports.append(
            {
                "decision_time": decision_time.isoformat(),
                "next_time": next_time.isoformat(),
                "selected_count": len([key for key in target_weights if key != "CASH"]),
                "gross_return": gross_return,
                "cost_return": cost_return,
                "cost_amount": cost_amount,
                "net_return": net_return,
                "pnl": pnl,
                "nav": nav,
                "turnover": turnover,
                "drawdown": drawdown,
                "target_weights": target_weights,
            }
        )
        previous_weights = target_weights

    if not period_reports:
        return _empty_backtest_report(
            effective_config,
            status="NO_DATA",
            reason="no tradable periods after lookback and alignment",
        )

    losing_abs = abs(losing_pnl)
    profit_factor: float | None = None if losing_abs == 0 else winning_pnl / losing_abs
    period_pnls = [item["pnl"] for item in period_reports]
    positive_periods = len([value for value in period_pnls if value > 0])
    negative_periods = len([value for value in period_pnls if value < 0])
    average_win = winning_pnl / positive_periods if positive_periods else 0.0
    average_loss = losing_pnl / negative_periods if negative_periods else 0.0
    cagr_value = _cagr(
        initial_nav=effective_config.initial_nav,
        final_nav=nav,
        start=all_timestamps[effective_config.lookback_bars],
        end=all_timestamps[-1],
    )
    annualized_volatility = _annualized_volatility(net_returns, effective_config.periods_per_year)
    sharpe = _sharpe(net_returns, effective_config.periods_per_year)
    sortino = _sortino(net_returns, effective_config.periods_per_year)
    calmar = (
        None
        if max_drawdown == 0 or cagr_value is None
        else cagr_value / abs(max_drawdown)
    )
    return {
        "schema": "multi_asset_rotation_backtest_v1",
        "status": "GO",
        "strategy": "offline_multi_asset_rotation",
        "config": effective_config.as_dict(),
        "asset_count": len(series),
        "instrument_count": len(series),
        "bar_count": sum(len(asset.timestamps) for asset in series),
        "period_count": len(period_reports),
        "rebalance_count": len(period_reports),
        "date_range": {
            "start": all_timestamps[0].isoformat(),
            "first_decision": all_timestamps[effective_config.lookback_bars].isoformat(),
            "end": all_timestamps[-1].isoformat(),
        },
        "initial_nav": effective_config.initial_nav,
        "final_nav": nav,
        "total_return": nav / effective_config.initial_nav - 1.0,
        "gross_return": total_gross_return,
        "net_return": nav / effective_config.initial_nav - 1.0,
        "CAGR": cagr_value,
        "annualized_volatility": annualized_volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "maximum_drawdown": max_drawdown,
        "max_drawdown": max_drawdown,
        "Calmar": calmar,
        "profit_factor": profit_factor,
        "profit_factor_status": _profit_factor_status(profit_factor),
        "gross_winning_pnl": winning_pnl,
        "gross_losing_pnl": losing_pnl,
        "expectancy": sum(period_pnls) / len(period_pnls),
        "win_rate": positive_periods / len(period_reports),
        "average_win": average_win,
        "average_loss": average_loss,
        "positive_periods": positive_periods,
        "negative_periods": negative_periods,
        "trade_count": trade_count,
        "average_turnover": total_turnover / len(period_reports),
        "turnover": total_turnover,
        "total_cost_return": total_cost_return,
        "total_costs": total_cost_amount,
        "cash_average": sum(cash_weights) / len(cash_weights),
        "cash_maximum": max(cash_weights),
        "sleeve_exposure": _average_exposures(sleeve_exposure_accumulator, len(period_reports)),
        "region_exposure": _average_exposures(region_exposure_accumulator, len(period_reports)),
        "last_target_weights": period_reports[-1]["target_weights"],
        "periods": period_reports,
        "research_warning": (
            "Positive PF is evidence only for this local dataset and configuration; "
            "it is not live-trading proof."
        ),
        "financial_calls": _zero_financial_calls(),
    }


def _build_signals(
    series: list[StrategyAssetSeries],
    *,
    all_timestamps: list[datetime],
    decision_time: datetime,
    config: MultiAssetBacktestConfig,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    decision_index = all_timestamps.index(decision_time)
    history_window = all_timestamps[decision_index - config.lookback_bars : decision_index + 1]
    for asset in series:
        prices = asset.close_by_timestamp()
        if decision_time not in prices:
            continue
        if any(timestamp not in prices for timestamp in history_window):
            continue
        closes = [prices[timestamp] for timestamp in history_window]
        returns = [current / previous - 1.0 for previous, current in zip(closes, closes[1:])]
        volatility = pstdev(returns) if len(returns) > 1 else 0.0
        volatility = max(volatility, 1e-6)
        raw_momentum = closes[-1] / closes[0] - 1.0
        trend_gate = closes[-1] > mean(closes)
        risk_adjusted_score = raw_momentum / volatility
        score = risk_adjusted_score if raw_momentum > 0 and trend_gate else 0.0
        if score <= config.min_score:
            continue
        signals.append(
            {
                "con_id": asset.con_id,
                "security_type": asset.security_type.value,
                "sleeve": asset.sleeve.value,
                "momentum": raw_momentum,
                "volatility": volatility,
                "trend_gate": trend_gate,
                "score": score,
            }
        )
    return signals


def _target_weights(
    signals: list[dict[str, Any]],
    config: MultiAssetBacktestConfig,
) -> dict[str, float]:
    target: dict[str, float] = {}
    used_weight = 0.0
    for sleeve, budget in config.sleeve_risk_budgets.items():
        if sleeve == SleeveName.DEFENSIVE_CASH or budget <= 0:
            continue
        sleeve_signals = sorted(
            [signal for signal in signals if signal["sleeve"] == sleeve.value],
            key=lambda signal: signal["score"],
            reverse=True,
        )[: config.top_n_per_sleeve]
        score_total = sum(signal["score"] for signal in sleeve_signals)
        if score_total <= 0:
            continue
        sleeve_used = 0.0
        for signal in sleeve_signals:
            raw_weight = budget * signal["score"] / score_total
            weight = min(raw_weight, config.max_asset_weight)
            target[str(signal["con_id"])] = target.get(str(signal["con_id"]), 0.0) + weight
            sleeve_used += weight
        used_weight += sleeve_used
    cash_weight = max(0.0, 1.0 - used_weight)
    target["CASH"] = cash_weight
    return target


def _one_way_turnover(previous_weights: dict[str, float], target_weights: dict[str, float]) -> float:
    keys = set(previous_weights) | set(target_weights)
    return 0.5 * sum(abs(target_weights.get(key, 0.0) - previous_weights.get(key, 0.0)) for key in keys)


def _sleeve_exposure(
    target_weights: dict[str, float],
    asset_by_id: dict[int, StrategyAssetSeries],
) -> dict[str, float]:
    exposures: dict[str, float] = {}
    for key, weight in target_weights.items():
        if key == "CASH":
            exposures[SleeveName.DEFENSIVE_CASH.value] = exposures.get(SleeveName.DEFENSIVE_CASH.value, 0.0) + weight
            continue
        asset = asset_by_id.get(int(key))
        sleeve = "UNKNOWN" if asset is None else asset.sleeve.value
        exposures[sleeve] = exposures.get(sleeve, 0.0) + weight
    return exposures


def _region_exposure(
    target_weights: dict[str, float],
    asset_by_id: dict[int, StrategyAssetSeries],
) -> dict[str, float]:
    exposures: dict[str, float] = {}
    for key, weight in target_weights.items():
        region = "CASH" if key == "CASH" else _default_region(asset_by_id.get(int(key)))
        exposures[region] = exposures.get(region, 0.0) + weight
    return exposures


def _default_region(asset: StrategyAssetSeries | None) -> str:
    if asset is None:
        return "UNKNOWN"
    return "GLOBAL_OR_UNCLASSIFIED"


def _accumulate_exposures(accumulator: dict[str, float], exposures: dict[str, float]) -> None:
    for key, value in exposures.items():
        accumulator[key] = accumulator.get(key, 0.0) + value


def _average_exposures(accumulator: dict[str, float], periods: int) -> dict[str, float]:
    if periods <= 0:
        return {}
    return {key: value / periods for key, value in sorted(accumulator.items())}


def _cagr(*, initial_nav: float, final_nav: float, start: datetime, end: datetime) -> float | None:
    elapsed_days = (end - start).total_seconds() / 86_400.0
    if elapsed_days <= 0 or final_nav <= 0:
        return None
    years = elapsed_days / 365.25
    if years <= 0:
        return None
    return (final_nav / initial_nav) ** (1.0 / years) - 1.0


def _annualized_volatility(returns: list[float], periods_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    return pstdev(returns) * periods_per_year**0.5


def _sharpe(returns: list[float], periods_per_year: float) -> float | None:
    if len(returns) < 2:
        return None
    sigma = pstdev(returns)
    if sigma == 0:
        return None
    return mean(returns) / sigma * periods_per_year**0.5


def _sortino(returns: list[float], periods_per_year: float) -> float | None:
    if len(returns) < 2:
        return None
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = (sum(value * value for value in downside) / len(downside)) ** 0.5
    if downside_deviation == 0:
        return None
    return mean(returns) / downside_deviation * periods_per_year**0.5


def _empty_backtest_report(
    config: MultiAssetBacktestConfig,
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": "multi_asset_rotation_backtest_v1",
        "status": status,
        "strategy": "offline_multi_asset_rotation",
        "reason": reason,
        "config": config.as_dict(),
        "asset_count": 0,
        "period_count": 0,
        "financial_calls": _zero_financial_calls(),
    }


def _profit_factor_status(profit_factor: float | None) -> str:
    if profit_factor is None:
        return "NO_LOSING_PERIODS"
    if profit_factor > 1.0:
        return "POSITIVE"
    if profit_factor == 1.0:
        return "BREAKEVEN"
    return "NEGATIVE"


def _default_sleeve_for_security_type(security_type: IbkrSecurityType) -> SleeveName:
    if security_type == IbkrSecurityType.FUT:
        return SleeveName.COMMODITY_TREND
    return SleeveName.ETF_CORE_ROTATION


def _parse_datetime(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _zero_financial_calls() -> dict[str, int]:
    return {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }
