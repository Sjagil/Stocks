from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    name: str
    weight: float
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("factor name is required")
        if self.weight <= 0:
            raise ValueError("factor weight must be positive")


DEFAULT_FACTOR_SPECS = (
    FactorSpec("momentum", 0.30),
    FactorSpec("quality", 0.25),
    FactorSpec("value", 0.20),
    FactorSpec("volatility", 0.15, higher_is_better=False),
    FactorSpec("liquidity", 0.10),
)


class CrossSectionalFactorEngine:
    """Point-in-time cross-sectional normalization and composite ranking."""

    def __init__(
        self,
        specs: Iterable[FactorSpec] = DEFAULT_FACTOR_SPECS,
        *,
        winsor_limits: tuple[float, float] = (0.01, 0.99),
    ):
        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("at least one factor is required")
        if len({spec.name for spec in self.specs}) != len(self.specs):
            raise ValueError("factor names must be unique")
        lower, upper = winsor_limits
        if not 0 <= lower < upper <= 1:
            raise ValueError("winsor_limits must satisfy 0 <= lower < upper <= 1")
        self.winsor_limits = winsor_limits

    def rank(
        self,
        snapshot: pd.DataFrame,
        *,
        as_of: Any,
        sector_neutral: bool = False,
        minimum_coverage: float = 1.0,
    ) -> pd.DataFrame:
        if not 0 < minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be in (0, 1]")
        required = {"symbol", "available_at", *(spec.name for spec in self.specs)}
        missing = sorted(required - set(snapshot.columns))
        if missing:
            raise ValueError(f"missing factor columns: {', '.join(missing)}")
        cutoff = pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        data = snapshot.copy()
        data["symbol"] = data["symbol"].astype("string").str.strip().str.upper()
        data["available_at"] = pd.to_datetime(data["available_at"], utc=True, errors="coerce")
        data = data.loc[data["available_at"].notna() & (data["available_at"] <= cutoff)]
        data = data.sort_values(["symbol", "available_at"], kind="stable").drop_duplicates("symbol", keep="last")
        if data.empty:
            return pd.DataFrame(columns=["rank", "symbol", "score", "percentile"])
        factor_names = [spec.name for spec in self.specs]
        for name in factor_names:
            data[name] = pd.to_numeric(data[name], errors="coerce")
            data.loc[~np.isfinite(data[name]), name] = np.nan
        data["factor_coverage"] = data[factor_names].notna().mean(axis=1)
        data = data.loc[data["factor_coverage"] >= minimum_coverage].copy()
        if data.empty:
            return pd.DataFrame(columns=["rank", "symbol", "score", "percentile"])

        weighted = pd.Series(0.0, index=data.index)
        active_weight = pd.Series(0.0, index=data.index)
        z_columns: list[str] = []
        for spec in self.specs:
            values = data[spec.name]
            lower, upper = values.quantile(list(self.winsor_limits))
            clipped = values.clip(lower=lower, upper=upper)
            zscore = _robust_zscore(clipped)
            if sector_neutral:
                if "sector" not in data:
                    raise ValueError("sector column is required for sector-neutral ranking")
                zscore = zscore - zscore.groupby(data["sector"].fillna("UNKNOWN")).transform("mean")
            if not spec.higher_is_better:
                zscore = -zscore
            column = f"{spec.name}_zscore"
            data[column] = zscore
            z_columns.append(column)
            available = zscore.notna()
            weighted = weighted.add(zscore.fillna(0.0) * spec.weight, fill_value=0.0)
            active_weight = active_weight.add(available.astype(float) * spec.weight, fill_value=0.0)
        data["score"] = weighted / active_weight.replace(0.0, np.nan)
        data = data.dropna(subset=["score"]).sort_values(["score", "symbol"], ascending=[False, True])
        data["rank"] = np.arange(1, len(data) + 1)
        data["percentile"] = data["score"].rank(method="average", pct=True) * 100.0
        columns = [
            "rank",
            "symbol",
            "score",
            "percentile",
            "factor_coverage",
            "available_at",
            *(["sector"] if "sector" in data else []),
            *factor_names,
            *z_columns,
        ]
        return data.loc[:, columns].reset_index(drop=True)


def build_market_factor_snapshot(
    bars: pd.DataFrame,
    *,
    as_of: Any,
    fundamentals: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Build momentum, volatility, liquidity, size, value and quality inputs.

    Price history is expected in long form with symbol/timestamp/close/volume.
    Fundamental rows are point-in-time and may contain earnings_yield,
    fcf_yield, roe, roic, gross_profitability, debt_to_equity, growth and
    market_cap.  Only information available at ``as_of`` is used.
    """

    required = {"symbol", "timestamp", "close", "volume"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"missing bar columns: {', '.join(missing)}")
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    data = bars.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if "available_at" in data:
        data["available_at"] = pd.to_datetime(data["available_at"], utc=True, errors="coerce")
        data = data.loc[data["available_at"] <= cutoff]
    data = data.loc[data["timestamp"] <= cutoff]
    rows: list[dict[str, Any]] = []
    for symbol, group in data.groupby(data["symbol"].astype("string").str.upper(), sort=True):
        group = group.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        close = pd.to_numeric(group["close"], errors="coerce").dropna()
        volume = pd.to_numeric(group.loc[close.index, "volume"], errors="coerce")
        if len(close) < 22 or (close <= 0).any():
            continue
        returns = close.pct_change(fill_method=None)
        row: dict[str, Any] = {
            "symbol": str(symbol),
            "available_at": cutoff,
            "momentum_12_1": _skip_month_momentum(close, 252, 21),
            "momentum_6m": _trailing_return(close, 126),
            "momentum_3m": _trailing_return(close, 63),
            "volatility": float(returns.tail(63).std(ddof=1) * np.sqrt(periods_per_year)),
            "liquidity": float(np.log1p((close * volume).tail(21).median())),
        }
        row["momentum"] = _nanmean([row["momentum_12_1"], row["momentum_6m"], row["momentum_3m"]])
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if fundamentals is not None and not fundamentals.empty:
        latest = _latest_fundamentals(fundamentals, cutoff)
        result = result.merge(latest, on="symbol", how="left", suffixes=("", "_fundamental"))
        result["value"] = result.apply(
            lambda row: _nanmean(
                [
                    row.get("earnings_yield"),
                    row.get("fcf_yield"),
                    _safe_inverse(row.get("ev_to_ebitda")),
                    _safe_inverse(row.get("price_to_book")),
                ]
            ),
            axis=1,
        )
        result["quality"] = result.apply(
            lambda row: _nanmean(
                [
                    row.get("roe"),
                    row.get("roic"),
                    row.get("gross_profitability"),
                    _safe_negative(row.get("debt_to_equity")),
                    _safe_negative(row.get("earnings_volatility")),
                ]
            ),
            axis=1,
        )
        result["size"] = -np.log1p(pd.to_numeric(result.get("market_cap"), errors="coerce"))
    else:
        result["value"] = np.nan
        result["quality"] = np.nan
        result["size"] = np.nan
    return result


def _latest_fundamentals(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    required = {"symbol", "available_at"}
    if not required <= set(frame):
        raise ValueError("fundamentals require symbol and available_at")
    data = frame.copy()
    data["symbol"] = data["symbol"].astype("string").str.upper()
    data["available_at"] = pd.to_datetime(data["available_at"], utc=True, errors="coerce")
    return (
        data.loc[data["available_at"] <= cutoff]
        .sort_values(["symbol", "available_at"])
        .drop_duplicates("symbol", keep="last")
    )


def _robust_zscore(values: pd.Series) -> pd.Series:
    mean = values.mean()
    standard_deviation = values.std(ddof=0)
    if pd.isna(standard_deviation) or standard_deviation <= 0:
        return pd.Series(0.0, index=values.index).where(values.notna())
    return (values - mean) / standard_deviation


def _trailing_return(values: pd.Series, periods: int) -> float:
    if len(values) <= periods:
        return np.nan
    return float(values.iloc[-1] / values.iloc[-periods - 1] - 1.0)


def _skip_month_momentum(values: pd.Series, lookback: int, skip: int) -> float:
    if len(values) <= lookback:
        return np.nan
    return float(values.iloc[-skip - 1] / values.iloc[-lookback - 1] - 1.0)


def _nanmean(values: Iterable[Any]) -> float:
    numbers = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    finite = numbers[np.isfinite(numbers)]
    return float(finite.mean()) if finite.size else np.nan


def _safe_inverse(value: Any) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(1.0 / number) if pd.notna(number) and number > 0 else np.nan


def _safe_negative(value: Any) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(-number) if pd.notna(number) else np.nan
