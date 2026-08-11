from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from stocks.execution.idempotency import stable_hash
from stocks.portfolio.p1_contracts import DesiredPortfolioTarget


PRIVATE_PATH = Path("data/portfolio/private/desired-portfolio-targets.json")
PUBLIC_PATH = Path("output/portfolio/desired-portfolio-targets.json")


def build_desired_portfolio_targets(
    project_root: Path,
    *,
    current_positions: Iterable[dict[str, Any]],
    sizing_rows: Iterable[dict[str, Any]],
    normalized_opportunities: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    current = {
        str(row.get("symbol") or "").upper(): row
        for row in current_positions
    }
    opportunity = {
        str(row.get("symbol") or "").upper(): row
        for row in normalized_opportunities.get("combined_ranking", [])
    }
    targets: list[dict[str, Any]] = []
    for row in sizing_rows:
        symbol = str(row.get("ticker") or "").upper()
        if not symbol:
            continue
        desired = _whole(row.get("target_quantity"))
        current_quantity = _number(row.get("current_quantity"))
        delta = float(desired) - current_quantity
        if desired == 0 and current_quantity > 0:
            action = "EXIT"
        elif delta > 0:
            action = "BUY_DELTA"
        elif delta < 0:
            action = "SELL_DELTA"
        elif current_quantity > 0:
            action = "HOLD"
        else:
            action = "NO_ACTION"
        candidate = opportunity.get(symbol, {})
        constraints = tuple(
            sorted(
                {
                    *(
                        str(value)
                        for value in row.get("execution_blockers", [])
                    ),
                    *(
                        str(value)
                        for value in candidate.get("blockers", [])
                    ),
                    *(
                        str(value)
                        for value in candidate.get(
                            "execution_blockers", []
                        )
                    ),
                    "EXECUTION_AUTHORITY_NONE",
                }
            )
        )
        target = DesiredPortfolioTarget(
            instrument_id=str(
                candidate.get("instrument_id")
                or current.get(symbol, {}).get("con_id")
                or symbol
            ),
            symbol=symbol,
            asset_class=str(candidate.get("asset_class") or "UNKNOWN"),
            desired_quantity=desired,
            desired_exposure_eur=_number(row.get("actual_notional_eur")),
            current_quantity=current_quantity,
            quantity_delta=delta,
            action=action,
            reason=str(
                row.get("whole_share_feasibility_status")
                or "NATIVE_RISK_TARGET"
            ),
            strategy_source=str(
                candidate.get("strategy_family") or "NATIVE_PORTFOLIO"
            ),
            confidence=_number(candidate.get("confidence")),
            expected_net_return=_optional(
                candidate.get("expected_net_return")
            ),
            expected_loss=_optional(candidate.get("expected_loss")),
            priority=_priority(candidate),
            expiry=candidate.get("signal_expiry"),
            constraints=constraints,
            rebalance_threshold_eur=float(
                policy.get("target_layer", {}).get(
                    "minimum_rebalance_notional_eur", 5.0
                )
            ),
            entry_reference=_optional(row.get("reference_price")),
            stop_price=_optional(row.get("stop_price")),
            take_profit_price=_optional(row.get("take_profit_price")),
            currency=(
                str(row.get("currency")) if row.get("currency") else None
            ),
            fx_rate_to_eur=_optional(row.get("fx_to_eur")),
        )
        targets.append(target.as_dict())
    targets.sort(key=lambda row: (-float(row["priority"]), row["symbol"]))
    rotations = evaluate_rotations(
        current_positions=current.values(),
        opportunities=normalized_opportunities.get(
            "combined_ranking", []
        ),
        policy=policy,
    )
    private: dict[str, Any] = {
        "schema": "desired_portfolio_target_book_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "current_position_count": len(current),
        "desired_target_count": len(targets),
        "targets": targets,
        "rotation_decisions": rotations,
        "submits_orders": False,
        "whole_share_risk_engine_authoritative": True,
        "execution_authority": "NONE",
        "orders_generated": 0,
        "broker_calls": 0,
    }
    private["content_hash"] = stable_hash(private)
    public_targets = [
        {
            "instrument_id": row["instrument_id"],
            "symbol": row["symbol"],
            "asset_class": row["asset_class"],
            "action": row["action"],
            "reason": row["reason"],
            "strategy_source": row["strategy_source"],
            "confidence": row["confidence"],
            "expected_net_return": row["expected_net_return"],
            "expected_loss": row["expected_loss"],
            "priority": row["priority"],
            "constraints": row["constraints"],
            "quantities_public": False,
            "financial_values_public": False,
            "execution_authority": "NONE",
        }
        for row in targets
    ]
    public = {
        **{
            key: value
            for key, value in private.items()
            if key not in {"targets", "content_hash"}
        },
        "targets": public_targets,
        "private_target_reference": PRIVATE_PATH.as_posix(),
    }
    public["content_hash"] = stable_hash(public)
    _write(project_root, private, public)
    return public


def evaluate_rotations(
    *,
    current_positions: Iterable[dict[str, Any]],
    opportunities: Iterable[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in opportunities
        if row.get("asset_class") != "CASH"
        and row.get("research_eligible")
        and row.get("expected_net_return") is not None
    ]
    by_symbol = {str(row["symbol"]).upper(): row for row in rows}
    ranked = sorted(
        rows,
        key=lambda row: (
            -_number(row.get("expected_net_return")),
            -_number(row.get("confidence")),
        ),
    )
    threshold = float(
        policy.get("rotation", {}).get(
            "minimum_expected_net_return_improvement", 0.02
        )
    )
    score_threshold = float(
        policy.get("rotation", {}).get(
            "minimum_score_improvement",
            policy["ranking"].get("replacement_improvement", 0.12),
        )
    )
    decisions: list[dict[str, Any]] = []
    for position in current_positions:
        symbol = str(position.get("symbol") or "").upper()
        held = by_symbol.get(symbol)
        alternative = next(
            (row for row in ranked if str(row["symbol"]).upper() != symbol),
            None,
        )
        if held is None:
            action = "EXIT_REQUIRED"
            reason = "HELD_POSITION_HAS_NO_CURRENT_VALID_OPPORTUNITY"
            improvement = None
        elif alternative is None:
            action = "KEEP_CURRENT_POSITION"
            reason = "NO_QUALIFIED_REPLACEMENT"
            improvement = None
        else:
            improvement = _number(
                alternative.get("expected_net_return")
            ) - _number(held.get("expected_net_return"))
            score_improvement = _number(
                alternative.get("confidence")
            ) - _number(held.get("confidence"))
            same_cluster = alternative.get("correlation_cluster") == held.get(
                "correlation_cluster"
            )
            rotate = (
                improvement >= threshold
                and score_improvement >= score_threshold
                and not same_cluster
            )
            action = "ROTATE" if rotate else "KEEP_CURRENT_POSITION"
            reason = (
                "MATERIAL_NET_IMPROVEMENT_AFTER_HYSTERESIS"
                if rotate
                else "ROTATION_HYSTERESIS_NOT_CLEARED"
            )
        decisions.append(
            {
                "symbol": symbol,
                "action": action,
                "reason": reason,
                "replacement_symbol": (
                    None if alternative is None else alternative["symbol"]
                ),
                "expected_net_return_improvement": improvement,
                "minimum_required_improvement": threshold,
                "transaction_costs_included_in_expected_net": True,
                "correlation_cluster_change_required": True,
                "automatic_execution": False,
                "execution_authority": "NONE",
            }
        )
    return decisions


def _write(
    project_root: Path,
    private: dict[str, Any],
    public: dict[str, Any],
) -> None:
    private_path = project_root / PRIVATE_PATH
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps(private, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    public_path = project_root / PUBLIC_PATH
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _priority(row: dict[str, Any]) -> float:
    net = _number(row.get("expected_net_return"))
    confidence = _number(row.get("confidence"))
    loss = max(_number(row.get("expected_loss")), 0.01)
    return round(max(0.0, net) * confidence / loss, 8)


def _whole(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "build_desired_portfolio_targets",
    "evaluate_rotations",
]
