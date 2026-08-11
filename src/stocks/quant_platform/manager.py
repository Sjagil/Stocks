from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from stocks.quant_platform.allocation import DynamicMultiAssetAllocator, PortfolioExposureRiskEngine
from stocks.quant_platform.capabilities import capability_registry
from stocks.quant_platform.execution import TransactionCostModel
from stocks.quant_platform.factors import CrossSectionalFactorEngine
from stocks.quant_platform.regime import RuleBasedRegimeDetector, StrategyAllocationEngine


@dataclass
class FullQuantPortfolioManager:
    """Research decision orchestrator with an intentionally closed broker boundary."""

    capital: float
    risk_budget: float = 0.10
    safety_margin: float = 0.002
    limits: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    cost_model: TransactionCostModel = field(default_factory=TransactionCostModel)

    def __post_init__(self) -> None:
        if self.capital <= 0 or not 0 < self.risk_budget <= 1 or self.safety_margin < 0:
            raise ValueError("invalid portfolio manager capital or risk policy")

    def run(
        self,
        *,
        returns: pd.DataFrame,
        asset_features: pd.DataFrame,
        factor_snapshot: pd.DataFrame,
        metadata: pd.DataFrame,
        macro_features: Mapping[str, float],
        current_weights: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        current_weights = dict(current_weights or {})
        regime = RuleBasedRegimeDetector().classify(macro_features)
        strategy_allocation = StrategyAllocationEngine().allocate({regime["regime"]: 1.0})
        ranking = CrossSectionalFactorEngine().rank(factor_snapshot, as_of=pd.Timestamp.now(tz="UTC"))
        covariance = returns.cov() * 252
        allocation = DynamicMultiAssetAllocator().allocate(
            asset_features,
            covariance,
            risk_budget=self.risk_budget,
        )
        target_risk_weights = {
            asset: weight
            for asset, weight in allocation["weights"].items()
            if asset != "CASH"
        }
        risk = PortfolioExposureRiskEngine().analyze(
            target_risk_weights,
            returns,
            metadata,
            limits=self.limits,
        )
        ranking_score = ranking.set_index("symbol")["score"].to_dict() if not ranking.empty else {}
        proposals = []
        for asset, target_weight in sorted(target_risk_weights.items()):
            current_weight = float(current_weights.get(asset, 0.0))
            delta_weight = float(target_weight - current_weight)
            if abs(delta_weight) < 1e-6:
                continue
            price = float(asset_features.loc[asset, "price"])
            average_daily_volume = float(asset_features.loc[asset, "average_daily_volume"])
            volatility = float(asset_features.loc[asset, "volatility"])
            quantity = int(math.floor(abs(delta_weight) * self.capital / price))
            expected_alpha = float(asset_features.loc[asset, "expected_return"])
            compliance = bool(metadata.loc[asset, "compliance_eligible"]) if "compliance_eligible" in metadata else False
            if quantity > 0:
                costs = self.cost_model.estimate(
                    price=price,
                    quantity=quantity,
                    average_daily_volume=average_daily_volume,
                    volatility=volatility,
                )
                economic = self.cost_model.economically_executable(expected_alpha, costs, safety_margin=self.safety_margin)
            else:
                costs = None
                economic = False
            blockers = []
            if not risk["risk_approved"]:
                blockers.append("PORTFOLIO_RISK_LIMIT")
            if not compliance:
                blockers.append("COMPLIANCE_NOT_ELIGIBLE")
            if quantity < 1:
                blockers.append("WHOLE_QUANTITY_NOT_FEASIBLE")
            if not economic:
                blockers.append("ALPHA_DOES_NOT_CLEAR_COST_AND_MARGIN")
            proposals.append(
                {
                    "symbol": asset,
                    "side": "INCREASE" if delta_weight > 0 else "REDUCE",
                    "current_weight": current_weight,
                    "target_weight": float(target_weight),
                    "delta_weight": delta_weight,
                    "whole_quantity": quantity,
                    "factor_rank_score": ranking_score.get(asset),
                    "expected_alpha": expected_alpha,
                    "expected_costs": costs,
                    "status": "PAPER_VALIDATION_CANDIDATE" if not blockers else "REJECTED",
                    "blockers": blockers,
                    "order_created": False,
                }
            )
        return {
            "schema": "full_quant_portfolio_manager_decision_v1",
            "pipeline": [
                "MARKET_DATA",
                "MACRO_REGIME",
                "FACTOR_TECHNICAL_NEWS_SEC_ALPHA",
                "PORTFOLIO_ALLOCATION",
                "COST_LIQUIDITY",
                "RISK_COMPLIANCE",
                "PAPER_VALIDATION",
                "ATTRIBUTION_FEEDBACK",
            ],
            "regime": regime,
            "strategy_allocation": strategy_allocation,
            "factor_ranking": ranking.to_dict(orient="records"),
            "target_allocation": allocation,
            "portfolio_risk": risk,
            "proposals": proposals,
            "capabilities": capability_registry(),
            "risk_approval": risk["risk_approved"],
            "manual_approval_required": True,
            "submission_allowed": False,
            "automatic_submission": False,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "broker_writes": 0,
        }


def manager_feedback(
    decisions: pd.DataFrame,
    realized_returns: pd.Series,
) -> dict[str, Any]:
    if "symbol" not in decisions or "expected_alpha" not in decisions:
        raise ValueError("decisions require symbol and expected_alpha")
    expected = decisions.set_index("symbol")["expected_alpha"].astype(float)
    realized = pd.to_numeric(realized_returns, errors="coerce")
    aligned = pd.concat([expected.rename("expected"), realized.rename("realized")], axis=1).dropna()
    errors = aligned["realized"] - aligned["expected"]
    return {
        "observations": len(aligned),
        "mean_forecast_error": float(errors.mean()),
        "mean_absolute_error": float(errors.abs().mean()),
        "rank_information_coefficient": float(aligned["expected"].corr(aligned["realized"], method="spearman")),
        "model_update_automatic": False,
        "human_review_required": True,
        "execution_authority": "NONE",
    }
