from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, fcluster
from scipy.spatial.distance import squareform


class HierarchicalRiskParity:
    def allocate(self, returns: pd.DataFrame) -> dict[str, Any]:
        clean = returns.apply(pd.to_numeric, errors="coerce").dropna()
        if clean.shape[1] < 2 or len(clean) < 2:
            raise ValueError("HRP requires at least two assets and observations")
        covariance = clean.cov()
        correlation = clean.corr().clip(-1, 1)
        distance = np.sqrt(np.maximum((1 - correlation.to_numpy()) / 2, 0))
        tree = linkage(squareform(distance, checks=False), method="single")
        order = leaves_list(tree).tolist()
        ordered_assets = [str(clean.columns[index]) for index in order]
        weights = pd.Series(1.0, index=ordered_assets)
        clusters = [ordered_assets]
        while clusters:
            next_clusters: list[list[str]] = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                split = len(cluster) // 2
                left, right = cluster[:split], cluster[split:]
                left_variance = _cluster_variance(covariance, left)
                right_variance = _cluster_variance(covariance, right)
                alpha = 1 - left_variance / (left_variance + right_variance)
                weights[left] *= alpha
                weights[right] *= 1 - alpha
                next_clusters.extend([left, right])
            clusters = next_clusters
        weights = weights.reindex(clean.columns).fillna(0.0)
        weights /= weights.sum()
        return {
            "method": "HIERARCHICAL_RISK_PARITY",
            "weights": weights.to_dict(),
            "cluster_order": ordered_assets,
            "covariance": covariance.to_dict(),
            "correlation": correlation.to_dict(),
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


class BlackLittermanAllocator:
    def allocate(
        self,
        covariance: pd.DataFrame,
        market_weights: Mapping[str, float],
        *,
        views: pd.DataFrame | None = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        long_only: bool = True,
    ) -> dict[str, Any]:
        assets = [str(column) for column in covariance.columns]
        matrix = covariance.loc[assets, assets].to_numpy(dtype=float)
        market = np.asarray([float(market_weights.get(asset, 0.0)) for asset in assets])
        if not np.isclose(market.sum(), 1.0) or risk_aversion <= 0 or tau <= 0:
            raise ValueError("invalid market weights, risk_aversion or tau")
        equilibrium = risk_aversion * matrix @ market
        posterior = equilibrium.copy()
        if views is not None and not views.empty:
            required = {"asset", "expected_return", "confidence"}
            if not required <= set(views):
                raise ValueError("views require asset, expected_return and confidence")
            p = np.zeros((len(views), len(assets)))
            q = np.zeros(len(views))
            omega = np.zeros((len(views), len(views)))
            for row_index, (_, view) in enumerate(views.iterrows()):
                asset = str(view["asset"])
                if asset not in assets:
                    raise ValueError(f"unknown view asset: {asset}")
                p[row_index, assets.index(asset)] = 1.0
                relative_to = view.get("relative_to")
                if pd.notna(relative_to):
                    relative = str(relative_to)
                    if relative not in assets:
                        raise ValueError(f"unknown relative view asset: {relative}")
                    p[row_index, assets.index(relative)] = -1.0
                q[row_index] = float(view["expected_return"])
                confidence = float(view["confidence"])
                if not 0 < confidence <= 1:
                    raise ValueError("view confidence must be in (0, 1]")
                view_variance = float(p[row_index] @ (tau * matrix) @ p[row_index])
                omega[row_index, row_index] = max(view_variance * (1 - confidence) / confidence, 1e-12)
            tau_covariance_inverse = np.linalg.pinv(tau * matrix)
            omega_inverse = np.linalg.pinv(omega)
            posterior_covariance = np.linalg.pinv(tau_covariance_inverse + p.T @ omega_inverse @ p)
            posterior = posterior_covariance @ (tau_covariance_inverse @ equilibrium + p.T @ omega_inverse @ q)
        raw = np.linalg.pinv(risk_aversion * matrix) @ posterior
        if long_only:
            raw = np.clip(raw, 0.0, None)
        if np.isclose(np.abs(raw).sum(), 0):
            raise ValueError("Black-Litterman produced zero weights")
        weights = raw / raw.sum() if long_only else raw / np.abs(raw).sum()
        return {
            "method": "BLACK_LITTERMAN",
            "market_implied_returns": dict(zip(assets, equilibrium.tolist(), strict=True)),
            "posterior_expected_returns": dict(zip(assets, posterior.tolist(), strict=True)),
            "weights": dict(zip(assets, weights.tolist(), strict=True)),
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


class DynamicMultiAssetAllocator:
    def allocate(
        self,
        features: pd.DataFrame,
        covariance: pd.DataFrame,
        *,
        risk_budget: float = 0.10,
        cash_symbol: str = "CASH",
    ) -> dict[str, Any]:
        required = {"expected_return", "volatility", "momentum", "macro", "liquidity", "drawdown"}
        missing = sorted(required - set(features))
        if missing:
            raise ValueError(f"missing allocator features: {', '.join(missing)}")
        assets = [str(index) for index in features.index]
        risk_assets = [asset for asset in assets if asset != cash_symbol]
        values = features.loc[risk_assets].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any() or (values["volatility"] <= 0).any():
            raise ValueError("allocator features must be complete with positive volatility")
        score = (
            0.35 * _zscore(values["expected_return"])
            + 0.20 * _zscore(values["momentum"])
            + 0.15 * _zscore(values["macro"])
            + 0.10 * _zscore(values["liquidity"])
            - 0.10 * _zscore(values["volatility"])
            + 0.10 * _zscore(values["drawdown"])
        )
        positive = np.exp(score - score.max()) / values["volatility"]
        risk_weights = positive / positive.sum()
        matrix = covariance.loc[risk_assets, risk_assets].to_numpy(dtype=float)
        volatility = math.sqrt(max(float(risk_weights.to_numpy() @ matrix @ risk_weights.to_numpy()), 0.0))
        scale = min(risk_budget / volatility, 1.0) if volatility > 0 else 0.0
        weights = (risk_weights * scale).to_dict()
        weights[cash_symbol] = 1.0 - sum(weights.values())
        return {
            "method": "DYNAMIC_MULTI_ASSET",
            "weights": weights,
            "scores": score.to_dict(),
            "expected_portfolio_volatility": volatility * scale,
            "risk_budget": risk_budget,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


class PortfolioExposureRiskEngine:
    def analyze(
        self,
        weights: Mapping[str, float],
        returns: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        factor_loadings: pd.DataFrame | None = None,
        limits: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        assets = [asset for asset in weights if asset in returns.columns]
        vector = pd.Series({asset: float(weights[asset]) for asset in assets})
        portfolio_returns = returns[assets].mul(vector, axis=1).sum(axis=1)
        volatility = float(portfolio_returns.std(ddof=1) * math.sqrt(252))
        drawdown = (1 + portfolio_returns).cumprod()
        maximum_drawdown = float((drawdown / drawdown.cummax() - 1).min())
        meta = metadata.reindex(assets)
        exposures = {
            dimension: vector.groupby(meta[dimension].fillna("UNKNOWN")).sum().to_dict()
            for dimension in ("sector", "country", "currency")
            if dimension in meta
        }
        factor_exposure = (
            factor_loadings.reindex(index=assets).mul(vector, axis=0).sum().to_dict()
            if factor_loadings is not None
            else {}
        )
        concentration = float(np.square(vector).sum())
        average_volume = pd.to_numeric(meta.get("average_daily_volume"), errors="coerce") if "average_daily_volume" in meta else pd.Series(index=assets, dtype=float)
        liquidity_risk = float((vector.abs() / np.log1p(average_volume).replace(0, np.nan)).sum(skipna=True))
        correlation = returns[assets].corr().fillna(0.0)
        if len(assets) > 1:
            distance = np.sqrt(np.maximum((1 - correlation.to_numpy()) / 2, 0))
            tree = linkage(squareform(distance, checks=False), method="average")
            cluster_ids = fcluster(tree, t=min(3, len(assets)), criterion="maxclust")
            clusters = dict(zip(assets, cluster_ids.astype(int).tolist(), strict=True))
        else:
            clusters = {assets[0]: 1} if assets else {}
        breaches = _limit_breaches(exposures, factor_exposure, limits or {})
        return {
            "gross_exposure": float(vector.abs().sum()),
            "net_exposure": float(vector.sum()),
            "exposures": exposures,
            "factor_exposure": factor_exposure,
            "annualized_volatility": volatility,
            "beta": float(factor_exposure.get("market", np.nan)),
            "maximum_drawdown": maximum_drawdown,
            "correlation_clusters": clusters,
            "liquidity_risk": liquidity_risk,
            "concentration_hhi": concentration,
            "limit_breaches": breaches,
            "risk_approved": not breaches,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


def _cluster_variance(covariance: pd.DataFrame, assets: list[str]) -> float:
    matrix = covariance.loc[assets, assets]
    inverse_diagonal = 1 / np.diag(matrix.to_numpy(dtype=float)).clip(min=np.finfo(float).eps)
    weights = inverse_diagonal / inverse_diagonal.sum()
    return float(weights @ matrix.to_numpy(dtype=float) @ weights)


def _zscore(values: pd.Series) -> pd.Series:
    standard_deviation = values.std(ddof=0)
    return (values - values.mean()) / standard_deviation if standard_deviation > 0 else pd.Series(0.0, index=values.index)


def _limit_breaches(
    exposures: Mapping[str, Mapping[str, float]],
    factors: Mapping[str, float],
    limits: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    breaches: list[dict[str, Any]] = []
    for dimension, configured in limits.items():
        observed = factors if dimension == "factor" else exposures.get(dimension, {})
        for label, maximum in configured.items():
            value = abs(float(observed.get(label, 0.0)))
            if value > float(maximum):
                breaches.append({"dimension": dimension, "label": label, "observed": value, "maximum": float(maximum)})
    return breaches
