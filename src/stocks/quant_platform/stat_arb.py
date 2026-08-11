from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class StatisticalArbitrageEngine:
    def analyze_pair(self, left: pd.Series, right: pd.Series, *, window: int = 60) -> dict[str, Any]:
        aligned = pd.concat(
            [pd.to_numeric(left, errors="coerce").rename("left"), pd.to_numeric(right, errors="coerce").rename("right")],
            axis=1,
        ).dropna()
        if len(aligned) < max(window, 40) or (aligned <= 0).any().any():
            raise ValueError("pair analysis requires sufficient positive aligned prices")
        x = np.column_stack([np.ones(len(aligned)), aligned["right"].to_numpy(dtype=float)])
        intercept, hedge_ratio = np.linalg.lstsq(x, aligned["left"].to_numpy(dtype=float), rcond=None)[0]
        spread = aligned["left"] - intercept - hedge_ratio * aligned["right"]
        adf_statistic, adf_pvalue, *_ = adfuller(spread, autolag="AIC")
        cointegration_statistic, cointegration_pvalue, _ = coint(aligned["left"], aligned["right"])
        johansen = coint_johansen(np.log(aligned), det_order=0, k_ar_diff=1)
        rolling_mean = spread.rolling(window).mean()
        rolling_standard_deviation = spread.rolling(window).std(ddof=1)
        zscore = (spread - rolling_mean) / rolling_standard_deviation.replace(0.0, np.nan)
        half_life = _half_life(spread)
        hurst = _hurst_exponent(spread)
        latest = float(zscore.iloc[-1])
        signal = "SHORT_SPREAD" if latest > 2 else "LONG_SPREAD" if latest < -2 else "EXIT_OR_HOLD"
        return {
            "observations": len(aligned),
            "hedge_ratio": float(hedge_ratio),
            "intercept": float(intercept),
            "adf_statistic": float(adf_statistic),
            "adf_pvalue": float(adf_pvalue),
            "engle_granger_statistic": float(cointegration_statistic),
            "engle_granger_pvalue": float(cointegration_pvalue),
            "johansen_trace_statistic": float(johansen.lr1[0]),
            "johansen_95pct_critical_value": float(johansen.cvt[0, 1]),
            "half_life": half_life,
            "hurst_exponent": hurst,
            "latest_spread": float(spread.iloc[-1]),
            "latest_zscore": latest,
            "research_signal": signal,
            "cointegrated_5pct": bool(adf_pvalue < 0.05 and cointegration_pvalue < 0.05),
            "execution_authority": "NONE",
            "broker_writes": 0,
        }

    def pca_residuals(self, returns: pd.DataFrame, *, components: int = 3) -> pd.DataFrame:
        clean = returns.apply(pd.to_numeric, errors="coerce").dropna()
        if not 1 <= components < clean.shape[1]:
            raise ValueError("components must be between 1 and asset_count - 1")
        standardized = (clean - clean.mean()) / clean.std(ddof=0).replace(0.0, np.nan)
        if standardized.isna().any().any():
            raise ValueError("constant assets cannot be PCA-normalized")
        pca = PCA(n_components=components, random_state=42)
        common = pca.inverse_transform(pca.fit_transform(standardized))
        return pd.DataFrame(standardized.to_numpy() - common, index=clean.index, columns=clean.columns)

    def cross_sectional_mean_reversion(
        self,
        returns: pd.Series,
        sectors: pd.Series,
        *,
        gross_exposure: float = 1.0,
    ) -> pd.DataFrame:
        frame = pd.concat(
            [pd.to_numeric(returns, errors="coerce").rename("return"), sectors.rename("sector")],
            axis=1,
        ).dropna()
        frame["market_adjusted"] = frame["return"] - frame["return"].mean()
        frame["sector_adjusted"] = frame["return"] - frame.groupby("sector")["return"].transform("mean")
        standard_deviation = frame["sector_adjusted"].std(ddof=0)
        frame["zscore"] = frame["sector_adjusted"] / standard_deviation if standard_deviation > 0 else 0.0
        raw = -frame["zscore"]
        raw = raw - raw.groupby(frame["sector"]).transform("mean")
        denominator = raw.abs().sum()
        frame["weight"] = raw * gross_exposure / denominator if denominator > 0 else 0.0
        return frame


def _half_life(spread: pd.Series) -> float | None:
    lagged = spread.shift(1).dropna()
    delta = spread.diff().reindex(lagged.index)
    x = np.column_stack([np.ones(len(lagged)), lagged.to_numpy(dtype=float)])
    coefficient = float(np.linalg.lstsq(x, delta.to_numpy(dtype=float), rcond=None)[0][1])
    return float(-math.log(2) / coefficient) if coefficient < 0 else None


def _hurst_exponent(series: pd.Series) -> float | None:
    values = series.to_numpy(dtype=float)
    lags = np.arange(2, min(100, len(values) // 2))
    if len(lags) < 5:
        return None
    tau = np.asarray([np.std(values[lag:] - values[:-lag]) for lag in lags])
    valid = tau > 0
    if valid.sum() < 5:
        return None
    return float(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0])
