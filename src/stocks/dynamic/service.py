from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.dynamic.strategy_allocation import (
    infer_strategy_family,
    score_strategy_evidence,
)
from stocks.notifications.telegram import TelegramQueue, telegram_daily_delivery
from stocks.portfolio.manager import build_active_portfolio_report
from stocks.research.autopilot.ledger import AutopilotLayout, ResearchLedger
from stocks.signals.service import signal_scan

REGIME_MATRIX_VERSION = "dynamic_regime_matrix_v1"
REGIMES = (
    "BULL_TREND_LOW_VOL",
    "BULL_TREND_HIGH_VOL",
    "SIDEWAYS_LOW_VOL",
    "SIDEWAYS_HIGH_VOL",
    "BEAR_TREND",
    "CRISIS",
    "RECOVERY",
    "INFLATIONARY_COMMODITY",
    "DEFENSIVE",
    "UNKNOWN",
)

FAMILY_MATRIX: dict[str, dict[str, float]] = {
    "ma_crossover": {
        "BULL_TREND_LOW_VOL": 1.0, "BULL_TREND_HIGH_VOL": 0.85,
        "RECOVERY": 0.85, "BEAR_TREND": 0.0, "CRISIS": 0.0,
    },
    "asymmetric_ma_crossover": {
        "BULL_TREND_LOW_VOL": 1.0, "BULL_TREND_HIGH_VOL": 0.9,
        "RECOVERY": 0.9, "BEAR_TREND": 0.0, "CRISIS": 0.0,
    },
    "ma_channel": {
        "BULL_TREND_LOW_VOL": 0.9, "BULL_TREND_HIGH_VOL": 0.75,
        "RECOVERY": 1.0, "SIDEWAYS_LOW_VOL": 0.3, "CRISIS": 0.0,
    },
    "bollinger_breakout": {
        "SIDEWAYS_LOW_VOL": 1.0, "RECOVERY": 1.0,
        "BULL_TREND_LOW_VOL": 0.7, "BULL_TREND_HIGH_VOL": 0.5,
        "CRISIS": 0.0,
    },
    "volatility_contraction_breakout": {
        "SIDEWAYS_LOW_VOL": 1.0, "RECOVERY": 1.0,
        "BULL_TREND_LOW_VOL": 0.7, "CRISIS": 0.0,
    },
    "cross_sectional_momentum": {
        "BULL_TREND_LOW_VOL": 1.0, "RECOVERY": 0.9,
        "BULL_TREND_HIGH_VOL": 0.65, "BEAR_TREND": 0.0, "CRISIS": 0.0,
    },
    "time_series_momentum": {
        "BULL_TREND_LOW_VOL": 1.0, "BULL_TREND_HIGH_VOL": 0.9,
        "BEAR_TREND": 0.7, "INFLATIONARY_COMMODITY": 0.9, "CRISIS": 0.2,
    },
    "quality_momentum": {
        "BULL_TREND_LOW_VOL": 0.9, "RECOVERY": 0.65,
        "DEFENSIVE": 0.7, "CRISIS": 0.0,
    },
    "etf_rotation": {
        "BULL_TREND_LOW_VOL": 0.9, "RECOVERY": 1.0,
        "DEFENSIVE": 1.0, "BEAR_TREND": 0.75, "CRISIS": 0.6,
    },
    "commodity_trend": {
        "INFLATIONARY_COMMODITY": 1.0, "BULL_TREND_HIGH_VOL": 0.8,
        "BEAR_TREND": 0.65, "CRISIS": 0.4,
    },
    "short_term_mean_reversion": {
        "SIDEWAYS_LOW_VOL": 0.85, "SIDEWAYS_HIGH_VOL": 0.7,
        "BULL_TREND_LOW_VOL": 0.45, "BEAR_TREND": 0.0, "CRISIS": 0.0,
    },
}

SELECTED_IDS = (
    "REC-21D39AD546399A65E47B",
    "REC-26626BAA352CB9DD6F9D",
    "REC-17ADF21FF74E9E8948A4",
    "REC-4F0C0A49849488D8367A",
    "REC-080A08D0A22C56FF305F",
    "REC-2E521DB7E46C1C8E141B",
    "DYN-P119-DONCHIAN-1W",
)


def dynamic_command(project_root: Path, command: str, *, symbol: str | None = None) -> dict[str, Any]:
    if command == "daily":
        return _run(project_root, refresh_signals=True)
    report = _run(project_root, refresh_signals=False)
    if command == "status":
        return report["status"]
    if command == "regime":
        return report["regime"]
    if command == "strategies":
        return {"status": "GO", "strategies": report["strategies"]}
    if command == "signals":
        return report["signals"]
    if command == "portfolio":
        return report["portfolio"]
    if command == "paper-campaign":
        return report["paper_campaign"]
    if command == "explain":
        name = str(symbol or "").upper()
        item = next((row for row in report["signals"]["signals"] if row["ticker"] == name), None)
        return {
            "status": "GO" if item else "NO_CURRENT_SIGNAL",
            "symbol": name,
            "decision": item,
            "regime": report["regime"]["regime"],
            "execution_authority": "NONE",
        }
    raise ValueError(f"UNKNOWN_DYNAMIC_COMMAND:{command}")


def classify_regime(close: pd.Series) -> dict[str, Any]:
    values = close.dropna().astype(float)
    if len(values) < 200:
        return _regime_payload("UNKNOWN", {}, ["INSUFFICIENT_200_SESSION_HISTORY"])
    last = float(values.iloc[-1])
    ma100 = float(values.tail(100).mean())
    ma200 = float(values.tail(200).mean())
    slope = ma200 / float(values.iloc[-220]) - 1 if len(values) >= 220 else 0.0
    returns = values.pct_change().dropna()
    vol20 = float(returns.tail(20).std(ddof=0) * math.sqrt(252))
    historical = returns.rolling(20).std(ddof=0).dropna() * math.sqrt(252)
    vol_pct = float((historical <= vol20).mean()) if len(historical) else 0.5
    drawdown = last / float(values.cummax().iloc[-1]) - 1
    if drawdown <= -0.20 and vol_pct >= 0.85:
        regime = "CRISIS"
    elif last < ma200 and slope < 0:
        regime = "BEAR_TREND"
    elif last > ma200 and slope > 0 and drawdown > -0.08:
        regime = "BULL_TREND_HIGH_VOL" if vol_pct >= 0.65 else "BULL_TREND_LOW_VOL"
    elif last > ma100 and slope > -0.02:
        regime = "RECOVERY"
    else:
        regime = "SIDEWAYS_HIGH_VOL" if vol_pct >= 0.65 else "SIDEWAYS_LOW_VOL"
    inputs = {
        "last_close": last, "ma100": ma100, "ma200": ma200,
        "ma200_slope": slope, "annualized_volatility_20": vol20,
        "volatility_percentile": vol_pct, "drawdown": drawdown,
    }
    return _regime_payload(regime, inputs, [])


def strategy_score(
    survivor: dict[str, Any], regime: str, *, data_quality: float = 1.0,
    forward_health: float | None = None, execution_quality: float | None = None,
    diversification_value: float | None = None,
) -> dict[str, Any]:
    del forward_health, execution_quality, diversification_value
    family = str(
        survivor.get("family")
        or infer_strategy_family(
            str(survivor.get("formula") or survivor.get("strategy_name") or "")
        )
    )
    compatibility = FAMILY_MATRIX.get(family, {}).get(regime, 0.4)
    evidence = score_strategy_evidence(
        survivor,
        regime_fit=compatibility,
        data_quality=data_quality,
    )
    components = {
        "historical_robustness": evidence["base_score"],
        "regime_compatibility": round(compatibility, 6),
        "data_quality": round(data_quality, 6),
        "sample_confidence": evidence["sample_confidence"],
        "evidence_penalty": evidence["evidence_penalty"],
    }
    return {
        "strategy_id": survivor.get("candidate_id") or survivor["strategy_id"],
        "family": family,
        "timeframe": survivor.get("timeframe", "1d"),
        "score": evidence["score"],
        "enabled": (
            compatibility > 0
            and data_quality > 0
            and evidence["evidence_status"] != "INSUFFICIENT_EVIDENCE"
        ),
        "components": components,
        "evidence": evidence,
        "frozen_parameters": _parameters(survivor.get("parameters")),
        "classification": survivor.get("classification", "FROZEN_SHADOW"),
        "financial_finalist": bool(survivor.get("financial_finalist", False)),
        "deployment_eligible": False,
        "execution_authority": "NONE",
    }


def capped_weights(
    scores: list[dict[str, Any]],
    cap: float = 0.25,
    family_cap: float = 0.50,
    temperature: float = 2.5,
) -> list[dict[str, Any]]:
    active = [row for row in scores if row["enabled"] and row["score"] >= 0.35]
    if active:
        maximum = max(float(row["score"]) for row in active)
        raw = {
            row["strategy_id"]: math.exp(
                temperature * (float(row["score"]) - maximum)
            )
            for row in active
        }
    else:
        raw = {}
    weights = _waterfill(raw, cap)
    by_family: dict[str, list[str]] = defaultdict(list)
    for row in active:
        by_family[row["family"]].append(row["strategy_id"])
    for ids in by_family.values():
        total = sum(weights.get(item, 0.0) for item in ids)
        if total > family_cap:
            factor = family_cap / total
            for item in ids:
                weights[item] *= factor
    weights = _redistribute_with_family_caps(
        weights,
        raw=raw,
        families={row["strategy_id"]: row["family"] for row in active},
        strategy_cap=cap,
        family_cap=family_cap,
    )
    return [
        {
            "strategy_id": row["strategy_id"], "family": row["family"],
            "weight": round(weights.get(row["strategy_id"], 0.0), 6),
            "enabled": row["enabled"],
            "score": row["score"],
            "temperature": temperature,
        }
        for row in scores
    ]


def _run(project_root: Path, *, refresh_signals: bool) -> dict[str, Any]:
    survivors = _read(project_root / "output/research/recovered_survivors.json").get("survivors", [])
    survivors.extend(
        _read(project_root / "config/dynamic/strategies_v1.json").get("strategies", [])
    )
    selected = [row for row in survivors if row.get("candidate_id") in SELECTED_IDS]
    selected.extend(_phase11_14_candidates(project_root))
    regime = _current_regime(project_root)
    forward_state = _read(
        project_root / "output/research/phase11_8/forward-holdout-registry.json"
    )
    scores = []
    for row in selected:
        scores.append(strategy_score(row, regime["regime"]))
    weights = capped_weights(scores)
    scan = signal_scan(project_root) if refresh_signals else _read(
        project_root / "output/signals/latest_signals.json"
    )
    signals = _consensus(project_root, scan.get("signals", []), weights, regime)
    portfolio = _portfolio(project_root, signals, regime)
    paper = _paper_campaign(project_root, portfolio)
    live = _live_readiness(project_root, portfolio)
    forward = forward_state
    strategy_registry = _strategy_registry(scores, weights)
    frozen_weekly = _read(
        project_root / "output/research/phase11_8/frozen/P118-AC6F644B3958E5C82EBF.json"
    ).get("candidate")
    if frozen_weekly:
        strategy_registry.append(
            {
                "strategy_id": frozen_weekly["candidate_id"],
                "family": "asymmetric_ma_crossover",
                "timeframe": "1w",
                "score": 0.0,
                "weight": 0.0,
                "enabled": False,
                "status": "FROZEN_FORWARD_OBSERVER",
                "reason": "FORWARD_DATA_OBSERVATION_ONLY_NOT_ACTIVE_VOTE",
                "frozen_parameters": {},
            }
        )
    phase119 = _read(project_root / "output/research/phase11_9/shortlist.json")
    for candidate in phase119.get("low_confidence_intraday_watchlist", []):
        strategy_registry.append(
            {
                "strategy_id": (
                    f"P119-{str(candidate['strategy']).upper()}-"
                    f"{str(candidate['timeframe']).upper()}"
                ),
                "family": candidate["strategy"],
                "timeframe": candidate["timeframe"],
                "score": 0.0,
                "weight": 0.0,
                "enabled": False,
                "status": "LOW_CONFIDENCE_OBSERVER",
                "reason": "ONLY_TWO_OOS_FOLDS_NOT_ELIGIBLE_FOR_DYNAMIC_VOTE",
                "evidence": candidate,
                "frozen_parameters": {},
            }
        )
    dynamic_forward = _dynamic_forward(
        project_root,
        strategy_registry,
        signals,
        append_observation=refresh_signals,
    )
    status = {
        "schema": "dynamic_multi_strategy_status_v1",
        "status": "DYNAMIC_MULTI_STRATEGY_ENGINE_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "current_regime": regime["regime"],
        "selected_strategy_count": len(selected),
        "active_timeframes": sorted(
            {
                str(row.get("timeframe", "1d"))
                for row in scores
                if row["enabled"]
            }
        ),
        "observed_timeframes": ["1d", "4h", "1w"],
        "blocked_timeframes": {
            "1h": "NO_STRATEGY_SURVIVED_50BPS_AND_SAMPLE_GATES",
        },
        "actionable_asset_count": len([x for x in signals["signals"] if x["action"] in {"BUY", "STRONG_BUY"}]),
        "portfolio_position_count": len(portfolio["positions"]),
        "DYNAMIC_MULTI_STRATEGY_ENGINE": True,
        "FROZEN_PARAMETERS": True,
        "REGIME_BASED_WEIGHTS": True,
        "BAYESIAN_EVIDENCE_UPDATING": True,
        "SOFTMAX_TEMPERATURE": 2.5,
        "STRATEGY_WEIGHT_CAP": 0.25,
        "STRATEGY_FAMILY_WEIGHT_CAP": 0.50,
        "ASSET_LEVEL_SIGNAL_NETTING": True,
        "WHOLE_SHARE_ACCOUNTING": True,
        "EUR_PORTFOLIO_ACCOUNTING": True,
        "MANUAL_SIGNALS_READY": True,
        "PAPER_CAMPAIGN_READY": paper["ready"],
        "LIVE_CANARY_READY": live["ready"],
        "CONTROLLED_LIVE_READY": False,
        "FORWARD_HOLDOUT_CONTINUES": True,
        "FORWARD_DATA_USED_FOR_RETUNING": False,
        "AUTOMATIC_LIVE_SCALING": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report: dict[str, Any] = {
        "status": status, "regime": regime, "strategies": strategy_registry,
        "signals": signals, "portfolio": portfolio, "paper_campaign": paper,
        "live_canary": live,
        "forward": {**forward, "dynamic": dynamic_forward},
        "telegram": {"status": "NOT_RUN"},
    }
    _publish(project_root, report)
    if refresh_signals:
        portfolio_manager = build_active_portfolio_report(project_root)
        allowed_ids = {
            str(row["signal_id"])
            for row in report["signals"]["signals"]
            if row["action"] in {"BUY", "STRONG_BUY"}
            and row.get("shariah_status")
            in {"SHARIAH_ELIGIBLE_PIT", "NOT_CONFIGURED"}
        }
        with TelegramQueue(project_root) as queue:
            quarantined = queue.quarantine_pending_signals_except(allowed_ids)
        report["telegram"] = telegram_daily_delivery(
            project_root, research={"status": "DYNAMIC_DAILY"}
        )
        report["telegram"]["portfolio_manager"] = {
            "status": portfolio_manager["status"],
            "content_hash": portfolio_manager["content_hash"],
            "execution_authority": "NONE",
        }
        report["telegram"]["quarantined_pending_signals"] = quarantined
        (project_root / "output/dynamic/telegram_status.json").write_text(
            json.dumps(report["telegram"], indent=2, default=str),
            encoding="utf-8",
        )
    return report


def _current_regime(project_root: Path) -> dict[str, Any]:
    path = project_root / "data/research/critical_trading/yfinance/SPY.parquet"
    if not path.exists():
        return _regime_payload("UNKNOWN", {}, ["SPY_REFERENCE_DATA_MISSING"])
    frame = pd.read_parquet(path).sort_values("session_date")
    payload = classify_regime(frame["close"])
    payload["data_timestamp"] = str(frame["session_date"].iloc[-1])
    payload["source"] = str(path.relative_to(project_root))
    return payload


def _regime_payload(regime: str, inputs: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    return {
        "schema": REGIME_MATRIX_VERSION, "status": "GO" if not blockers else "DEGRADED",
        "regime": regime, "as_of": datetime.now(UTC).isoformat(), "inputs": inputs,
        "blockers": blockers, "causal": True, "future_data_used": False,
    }


def _strategy_registry(scores: list[dict[str, Any]], weights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weight_map = {row["strategy_id"]: row["weight"] for row in weights}
    rows = [{**row, "weight": weight_map.get(row["strategy_id"], 0.0), "status": "FROZEN_DYNAMIC"} for row in scores]
    represented = {row["family"] for row in rows}
    for family in FAMILY_MATRIX:
        if family not in represented:
            rows.append({
                "strategy_id": f"FAMILY-{family.upper()}", "family": family,
                "timeframe": "1d_or_higher",
                "score": 0.0, "weight": 0.0, "enabled": False,
                "status": "DATA_LIMITED" if family == "quality_momentum" else "RESEARCH_ONLY",
                "reason": "NO_DISTINCT_FROZEN_SURVIVOR",
                "frozen_parameters": {},
            })
    return rows


def _consensus(
    project_root: Path,
    plans: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    regime: dict[str, Any],
) -> dict[str, Any]:
    weight_map = {row["strategy_id"]: float(row["weight"]) for row in weights}
    family_map = {row["strategy_id"]: row["family"] for row in weights}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for plan in plans:
        if plan.get("data_freshness") == "STALE":
            continue
        grouped[str(plan.get("ticker", "")).upper()].append(plan)
    rows = []
    for ticker, items in grouped.items():
        participating = [item for item in items if weight_map.get(item.get("strategy_id"), 0) > 0]
        if not participating:
            continue
        families = {family_map[item["strategy_id"]] for item in participating}
        blockers = {
            risk
            for item in participating
            for risk in item.get("risks", [])
            if risk in {"EARNINGS_EVENT_BLOCKER", "RISK_BLOCKER", "STALE_DATA"}
        }
        shariah = _shariah_status(project_root, ticker)
        if shariah == "SHARIAH_ATTESTATION_REQUIRED":
            blockers.add(shariah)
        family_contributions: dict[str, float] = defaultdict(float)
        for item in participating:
            family = family_map[item["strategy_id"]]
            contribution = weight_map[item["strategy_id"]] * float(item.get("confidence_score", 0))
            family_contributions[family] = max(family_contributions[family], contribution)
        score = sum(family_contributions.values())
        score = min(1.0, score)
        action = (
            "AVOID"
            if blockers
            else "STRONG_BUY"
            if score >= 0.55 and len(families) >= 2
            else "BUY"
            if score >= 0.20
            else "WATCHLIST"
        )
        base = max(participating, key=lambda item: float(item.get("confidence_score", 0)))
        dominant_family = max(
            family_contributions, key=family_contributions.__getitem__
        )
        risk_policy = _family_risk_policy(dominant_family)
        row = {
            **base, "action": action, "consensus_score": round(score, 6),
            "participating_strategies": [
                {"strategy_id": item["strategy_id"], "weight": weight_map[item["strategy_id"]],
                 "confidence": float(item.get("confidence_score", 0))}
                for item in participating
            ],
            "independent_family_count": len(families), "regime": regime["regime"],
            "family_contributions": {key: round(value, 6) for key, value in family_contributions.items()},
            "dominant_family": dominant_family,
            **risk_policy,
            "risk_blockers": sorted(blockers),
            "shariah_status": shariah,
            "automatic_execution_allowed": False, "execution_authority": "NONE",
        }
        rows.append(row)
    rows.sort(key=lambda row: (-row["consensus_score"], row["ticker"]))
    return {"schema": "dynamic_asset_consensus_v1", "status": "GO", "signals": rows, "execution_authority": "NONE"}


def _family_risk_policy(family: str) -> dict[str, str]:
    if family in {
        "bollinger_breakout",
        "volatility_contraction_breakout",
        "time_series_momentum",
    }:
        return {
            "stop_method": "BREAKOUT_LEVEL_WITH_ATR_BUFFER",
            "take_profit_mode": "TP1_1_5R_TP2_2_5R_TRAIL_REMAINDER",
            "exit_policy": "CLOSE_BACK_INSIDE_CHANNEL_OR_ATR_TRAIL_OR_TIME_STOP",
        }
    if family == "etf_rotation":
        return {
            "stop_method": "ABSOLUTE_TREND_INVALIDATION",
            "take_profit_mode": "NO_FIXED_FINAL_TARGET_TRAILING_ROTATION_EXIT",
            "exit_policy": "RELATIVE_RANK_OR_ABSOLUTE_TREND_EXIT_AT_REBALANCE",
        }
    if family == "short_term_mean_reversion":
        return {
            "stop_method": "STRUCTURAL_INVALIDATION_WITH_VOLATILITY_HARD_STOP",
            "take_profit_mode": "MEAN_REVERSION_TARGET_AND_SHORT_TIME_STOP",
            "exit_policy": "MEAN_REVERSION_OR_TIME_STOP",
        }
    return {
        "stop_method": "ATR_CHANDELIER_AND_MOVING_AVERAGE_INVALIDATION",
        "take_profit_mode": "TP1_1_5R_TP2_2_5R_TRAIL_REMAINDER",
        "exit_policy": "MOVING_AVERAGE_INVALIDATION_OR_ATR_TRAIL",
    }


def _dynamic_forward(
    project_root: Path,
    strategies: list[dict[str, Any]],
    signals: dict[str, Any],
    *,
    append_observation: bool,
) -> dict[str, Any]:
    active = [row for row in strategies if row.get("status") == "FROZEN_DYNAMIC"]
    ledger = ResearchLedger(AutopilotLayout(project_root))
    try:
        registrations = [
            ledger.register_dynamic_forward(
                {
                    "strategy_id": row["strategy_id"],
                    "family": row["family"],
                    "timeframe": row["timeframe"],
                    "frozen_parameters": row["frozen_parameters"],
                    "classification": row.get("classification"),
                    "forward_data_used_for_retuning": False,
                }
            )
            for row in active
        ]
        appended = []
        if append_observation:
            session_values = [
                str(row.get("data_timestamp", ""))[:10]
                for row in signals["signals"]
                if row.get("data_timestamp")
            ]
            session_date = max(session_values) if session_values else datetime.now(UTC).date().isoformat()
            for strategy in active:
                strategy_id = strategy["strategy_id"]
                assets = sorted(
                    {
                        row["ticker"]
                        for row in signals["signals"]
                        if any(
                            item["strategy_id"] == strategy_id
                            for item in row.get("participating_strategies", [])
                        )
                    }
                )
                appended.append(
                    ledger.append_dynamic_forward_observation(
                        strategy_id=strategy_id,
                        session_date=session_date,
                        payload={
                            "signal_count": len(assets),
                            "signal_assets": assets,
                            "parameters_mutated": False,
                            "used_for_optimization": False,
                            "order_intents": [],
                            "automatic_orders": 0,
                            "execution_authority": "NONE",
                        },
                    )
                )
        status = ledger.dynamic_forward_status()
    finally:
        ledger.close()
    return {
        "status": "GO",
        "registration_count": status["registration_count"],
        "observation_count": status["observation_count"],
        "registered_now": sum(row["inserted"] for row in registrations),
        "observations_appended_now": sum(row["inserted"] for row in appended),
        "frozen_hashes": {
            row["strategy_id"]: row["frozen_hash"] for row in registrations
        },
        "authority": "NONE",
        "forward_data_used_for_retuning": False,
    }


def _portfolio(
    project_root: Path,
    signals: dict[str, Any],
    regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capital = Decimal("10000")
    regime_name = str((regime or {}).get("regime", "UNKNOWN"))
    regime_multiplier = {
        "BULL_TREND_LOW_VOL": Decimal("1"),
        "BULL_TREND_HIGH_VOL": Decimal("0.70"),
        "SIDEWAYS_LOW_VOL": Decimal("0.60"),
        "SIDEWAYS_HIGH_VOL": Decimal("0.40"),
        "BEAR_TREND": Decimal("0.20"),
        "CRISIS": Decimal("0"),
    }.get(regime_name, Decimal("0.40"))
    drawdown = Decimal(
        str((regime or {}).get("inputs", {}).get("drawdown", "0"))
    )
    drawdown_multiplier = (
        Decimal("0")
        if drawdown <= Decimal("-0.20")
        else Decimal("0.50")
        if drawdown <= Decimal("-0.12")
        else Decimal("0.75")
        if drawdown <= Decimal("-0.08")
        else Decimal("1")
    )
    hmm = _hmm_risk_overlay(project_root)
    hmm_multiplier = Decimal(str(hmm["multiplier"]))
    exposure_target = (
        Decimal("0.25")
        * regime_multiplier
        * drawdown_multiplier
        * hmm_multiplier
    )
    max_exposure = capital * exposure_target
    positions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used = Decimal("0")
    ranked_signals = sorted(
        signals["signals"],
        key=lambda row: (
            -float(row.get("consensus_score", 0)),
            str(row.get("ticker", "")),
        ),
    )
    for row in ranked_signals:
        if row["action"] not in {"BUY", "STRONG_BUY"} or len(positions) >= 3:
            continue
        if not row.get("contract_identity"):
            rejected.append({"ticker": row["ticker"], "reason": "CONTRACT_IDENTITY_REQUIRED"})
            continue
        price = Decimal(str(row["preferred_entry"]))
        stop = Decimal(str(row["stop_loss"]))
        fx = _fx_to_eur(project_root, str(row.get("currency", "USD")))
        allowed_risk = (
            capital
            * Decimal("0.0025")
            * Decimal(str(row["consensus_score"]))
            * hmm_multiplier
        )
        risk_per_share = abs(price - stop) * fx
        risk_notional = (
            allowed_risk / risk_per_share * price * fx
            if risk_per_share > 0
            else Decimal("0")
        )
        score_weight = Decimal(str(max(float(row["consensus_score"]), 0.0)))
        stop_distance_pct = (
            risk_per_share / (price * fx)
            if price > 0 and fx > 0
            else Decimal("1")
        )
        inverse_risk_score = (
            score_weight / stop_distance_pct
            if stop_distance_pct > 0
            else Decimal("0")
        )
        score_cap = min(
            Decimal("1"),
            inverse_risk_score / Decimal("10"),
        )
        per_asset = min(
            capital * Decimal("0.05") * score_cap,
            max(Decimal("0"), max_exposure - used),
            risk_notional,
        )
        qty = (per_asset / (price * fx)).quantize(Decimal("1"), rounding=ROUND_DOWN)
        if qty <= 0:
            one_share_notional = price * fx
            required_risk = risk_per_share
            candidates.append(
                {
                    "ticker": row["ticker"],
                    "target_quantity": 0,
                    "proposal_status": "VALID_SIGNAL_BELOW_WHOLE_SHARE_BUDGET",
                    "status": "BELOW_WHOLE_SHARE_BUDGET",
                    "required_cash_eur": str(
                        one_share_notional.quantize(Decimal("0.01"))
                    ),
                    "required_risk_eur": str(
                        required_risk.quantize(Decimal("0.01"))
                    ),
                    "available_risk_eur": str(
                        allowed_risk.quantize(Decimal("0.01"))
                    ),
                    "sizing_mode": "RISK_SIZED_WHOLE_SHARE",
                    "execution_authority": "NONE",
                }
            )
            continue
        notional = qty * price * fx
        positions.append({
            "ticker": row["ticker"], "quantity": int(qty), "entry": str(row["preferred_entry"]),
            "stop_loss": str(row["stop_loss"]), "take_profit_1": str(row["take_profit_1"]),
            "take_profit_2": str(row["take_profit_2"]), "notional_eur": str(notional.quantize(Decimal("0.01"))),
            "consensus_score": row["consensus_score"], "execution_authority": "NONE",
        })
        candidates.append(
            {
                "ticker": row["ticker"],
                "target_quantity": int(qty),
                "proposal_status": "VALID_SIGNAL_EXECUTABLE",
                "status": "EXECUTABLE",
                "required_cash_eur": str(
                    notional.quantize(Decimal("0.01"))
                ),
                "required_risk_eur": str(
                    (qty * risk_per_share).quantize(Decimal("0.01"))
                ),
                "available_risk_eur": str(
                    allowed_risk.quantize(Decimal("0.01"))
                ),
                "sizing_mode": "RISK_SIZED_WHOLE_SHARE",
                "execution_authority": "NONE",
            }
        )
        used += notional
    if not candidates and not rejected:
        proposal_status = "NO_SIGNAL"
    elif positions:
        proposal_status = "VALID_SIGNAL_EXECUTABLE"
    elif candidates:
        proposal_status = "VALID_SIGNAL_BELOW_WHOLE_SHARE_BUDGET"
    else:
        proposal_status = "VALID_SIGNAL_BLOCKED_BY_RISK"
    return {
        "schema": "dynamic_portfolio_proposal_v2",
        "status": "GO",
        "proposal_status": proposal_status,
        "positions": positions,
        "candidates": candidates,
        "rejected_candidates": rejected,
        "gross_exposure_eur": str(used.quantize(Decimal("0.01"))),
        "gross_exposure_pct": float(used / capital), "cash_reserve_pct": float(1 - used / capital),
        "max_total_exposure_pct": float(exposure_target),
        "base_max_total_exposure_pct": 0.25,
        "regime": regime_name,
        "regime_exposure_multiplier": float(regime_multiplier),
        "drawdown_multiplier": float(drawdown_multiplier),
        "hmm_regime_status": hmm["status"],
        "hmm_regime_as_of": hmm["as_of"],
        "hmm_risk_multiplier": float(hmm_multiplier),
        "hmm_can_only_reduce_risk": True,
        "portfolio_formula": "SCORE_WEIGHTED_INVERSE_STOP_RISK",
        "selection_order": "CONSENSUS_SCORE_DESC_TICKER_ASC",
        "max_open_positions": 3,
        "max_sector_exposure_pct": 0.15, "max_region_exposure_pct": 0.20,
        "max_position_pct": 0.05, "base_risk_per_trade_pct": 0.25,
        "whole_share_accounting": True, "base_currency": "EUR", "execution_authority": "NONE",
    }


def _hmm_risk_overlay(project_root: Path) -> dict[str, Any]:
    payload = _read(
        project_root
        / "output"
        / "research"
        / "phase11_11"
        / "current.json"
    )
    if payload.get("status") != "GO":
        return {
            "status": "OPTIONAL_HMM_UNAVAILABLE",
            "as_of": None,
            "multiplier": 1.0,
        }
    state = payload.get("state", {})
    try:
        multiplier = float(state["regime_multiplier"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "INVALID_HMM_STATE_IGNORED",
            "as_of": state.get("as_of"),
            "multiplier": 1.0,
        }
    return {
        "status": "GO",
        "as_of": state.get("as_of"),
        "multiplier": min(max(multiplier, 0.0), 1.0),
    }


def _paper_campaign(project_root: Path, portfolio: dict[str, Any]) -> dict[str, Any]:
    phase9 = _read(project_root / "output/ibkr/phase9/status.json")
    blockers = list(phase9.get("open_blockers", []))
    return {
        "schema": "dynamic_paper_campaign_v1", "status": "READY" if not blockers else "BLOCKED",
        "ready": not blockers, "strategy_limit": 3, "asset_limit": 5, "position_limit": 3,
        "proposed_positions": len(portfolio["positions"]), "authority_required": "MANUAL_PAPER_CANARY",
        "execution_authority": "NONE", "blockers": blockers, "broker_calls": 0, "orders_generated": 0,
    }


def _live_readiness(project_root: Path, portfolio: dict[str, Any]) -> dict[str, Any]:
    phase9 = _read(project_root / "output/ibkr/phase9/status.json")
    blockers = list(phase9.get("open_blockers", []))
    if (
        not portfolio["positions"]
        and portfolio.get("proposal_status") != "VALID_SIGNAL_BELOW_WHOLE_SHARE_BUDGET"
    ):
        blockers.append("NO_CURRENT_PORTFOLIO_PROPOSAL")
    if portfolio.get("proposal_status") == "VALID_SIGNAL_BELOW_WHOLE_SHARE_BUDGET":
        blockers.append("CURRENT_SIGNAL_BELOW_WHOLE_SHARE_BUDGET")
    return {
        "schema": "dynamic_live_canary_readiness_v1", "status": "BLOCKED" if blockers else "OPERATOR_APPROVAL_REQUIRED",
        "ready": not blockers, "max_order_eur": 10, "max_total_exposure_eur": 25,
        "max_positions": 1, "max_new_orders_per_day": 1, "no_autoscaling": True,
        "execution_authority": "NONE", "blockers": sorted(set(blockers)),
    }


def _waterfill(raw: dict[str, float], cap: float) -> dict[str, float]:
    if not raw or sum(raw.values()) <= 0:
        return {key: 0.0 for key in raw}
    remaining = set(raw)
    result = {key: 0.0 for key in raw}
    budget = 1.0
    while remaining and budget > 1e-12:
        denominator = sum(raw[key] for key in remaining)
        if denominator <= 0:
            break
        capped = []
        for key in remaining:
            allocation = budget * raw[key] / denominator
            if allocation >= cap:
                result[key] = cap
                budget -= cap
                capped.append(key)
        if not capped:
            for key in remaining:
                result[key] = budget * raw[key] / denominator
            break
        remaining.difference_update(capped)
    return result


def _redistribute_with_family_caps(
    weights: dict[str, float],
    *,
    raw: dict[str, float],
    families: dict[str, str],
    strategy_cap: float,
    family_cap: float,
) -> dict[str, float]:
    result = dict(weights)
    for _ in range(20):
        family_totals: dict[str, float] = defaultdict(float)
        for strategy_id, weight in result.items():
            family_totals[families[strategy_id]] += weight
        changed = False
        for family, total in family_totals.items():
            if total <= family_cap + 1e-12:
                continue
            factor = family_cap / total
            for strategy_id in result:
                if families[strategy_id] == family:
                    result[strategy_id] *= factor
            changed = True
        budget = max(0.0, 1.0 - sum(result.values()))
        if budget <= 1e-12:
            break
        family_totals = defaultdict(float)
        for strategy_id, weight in result.items():
            family_totals[families[strategy_id]] += weight
        eligible = {
            strategy_id
            for strategy_id, weight in result.items()
            if weight < strategy_cap - 1e-12
            and family_totals[families[strategy_id]] < family_cap - 1e-12
        }
        if not eligible:
            break
        denominator = sum(raw[strategy_id] for strategy_id in eligible)
        if denominator <= 0:
            break
        distributed = 0.0
        for strategy_id in eligible:
            family_room = family_cap - family_totals[families[strategy_id]]
            strategy_room = strategy_cap - result[strategy_id]
            addition = min(
                budget * raw[strategy_id] / denominator,
                family_room,
                strategy_room,
            )
            result[strategy_id] += addition
            family_totals[families[strategy_id]] += addition
            distributed += addition
        if distributed <= 1e-12 and not changed:
            break
    return result


def _phase11_14_candidates(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "output/research/phase11_14/strategy-summary.parquet"
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    required = {
        "strategy_id",
        "robust_pass",
        "portfolio_invariants_go",
        "forward_observer_candidate",
    }
    if frame.empty or not required.issubset(frame.columns):
        return []
    selected = frame[
        frame["robust_pass"].astype(bool)
        & frame["portfolio_invariants_go"].astype(bool)
        & frame["forward_observer_candidate"].astype(bool)
    ]
    rows = []
    for _, item in selected.sort_values("strategy_id").iterrows():
        formula = str(item.get("formula") or "unknown")
        rows.append(
            {
                "candidate_id": str(item["strategy_id"]),
                "strategy_name": formula,
                "family": infer_strategy_family(formula),
                "timeframe": str(item.get("timeframe") or "1d"),
                "profit_factor": item.get("combined_period_profit_factor"),
                "sharpe": item.get("combined_oos_Sharpe"),
                "sample_count": item.get("normal_cost_fill_count"),
                "positive_periods": item.get("positive_fold_count"),
                "total_periods": item.get("fold_count"),
                "parameter_plateau_ratio": item.get("parameter_plateau_ratio"),
                "parameters": {
                    "qualification_strategy_id": str(item["strategy_id"]),
                    "source_strategy_id": str(item.get("source_strategy_id") or ""),
                    "frozen_profile": str(item.get("frozen_profile") or ""),
                },
                "classification": "FROZEN_SHADOW",
                "financial_finalist": bool(item.get("financial_finalist", False)),
                "evidence_scope": str(item.get("evidence_scope") or "UNKNOWN"),
                "deployment_blockers": str(item.get("deployment_blockers") or ""),
            }
        )
    return rows


def _parameters(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _fx_to_eur(project_root: Path, currency: str) -> Decimal:
    if currency.upper() == "EUR":
        return Decimal("1")
    path = project_root / "data/fx/fx_daily.parquet"
    if not path.exists():
        raise ValueError(f"FX_DATA_REQUIRED:{currency}")
    frame = pd.read_parquet(path)
    if {"base_currency", "quote_currency", "rate"}.issubset(frame.columns):
        rows = frame[
            frame["base_currency"].astype(str).str.upper().eq(currency.upper())
            & frame["quote_currency"].astype(str).str.upper().eq("EUR")
        ]
        if not rows.empty:
            value = Decimal(str(rows.iloc[-1]["rate"]))
            if value > 0:
                return value
    for currency_column in ("quote_currency", "currency"):
        if currency_column not in frame:
            continue
        rows = frame[frame[currency_column].astype(str).str.upper().eq(currency.upper())]
        if rows.empty:
            continue
        for rate_column in ("fx_to_eur", "rate_to_eur", "eur_rate"):
            if rate_column in rows:
                value = Decimal(str(float(rows[rate_column].iloc[-1])))
                if value > 0:
                    return value
    raise ValueError(f"FX_RATE_REQUIRED:{currency}")


def _shariah_status(project_root: Path, symbol: str) -> str:
    config = _read(project_root / "config/screener/daily_screener_v1.json")
    relative = config.get("shariah_attestations_path")
    if not relative:
        return "NOT_CONFIGURED"
    registry = _read(project_root / str(relative))
    now = datetime.now(UTC)
    for row in registry.get("attestations", []):
        if str(row.get("symbol", "")).upper() != symbol.upper():
            continue
        try:
            expires = datetime.fromisoformat(
                str(row["expires_at"]).replace("Z", "+00:00")
            )
        except (KeyError, ValueError):
            continue
        if (
            row.get("status") == "SHARIAH_ELIGIBLE_PIT"
            and expires >= now
        ):
            return "SHARIAH_ELIGIBLE_PIT"
    return "SHARIAH_ATTESTATION_REQUIRED"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _publish(project_root: Path, report: dict[str, Any]) -> None:
    root = project_root / "output/dynamic"
    root.mkdir(parents=True, exist_ok=True)
    allocated = [
        row
        for row in report["strategies"]
        if float(row.get("weight", 0.0)) > 0
    ]
    family_weights: dict[str, float] = defaultdict(float)
    for row in allocated:
        family_weights[str(row["family"])] += float(row["weight"])
    files = {
        "status.json": report["status"],
        "current_regime.json": report["regime"],
        "strategy_scores.json": {"strategies": report["strategies"]},
        "strategy_weights.json": {
            "schema": "dynamic_strategy_weights_v2",
            "status": "GO",
            "temperature": 2.5,
            "strategy_cap": 0.25,
            "family_cap": 0.50,
            "allocated_weight": round(
                sum(float(row["weight"]) for row in allocated), 6
            ),
            "unallocated_weight": round(
                max(0.0, 1.0 - sum(float(row["weight"]) for row in allocated)),
                6,
            ),
            "weights": [
                {
                    "strategy_id": row["strategy_id"],
                    "family": row["family"],
                    "timeframe": row.get("timeframe"),
                    "score": row.get("score"),
                    "weight": row.get("weight", 0.0),
                    "evidence_status": row.get("evidence", {}).get(
                        "evidence_status",
                        row.get("status"),
                    ),
                }
                for row in report["strategies"]
            ],
            "execution_authority": "NONE",
        },
        "strategy-allocation-audit.json": {
            "schema": "dynamic_strategy_allocation_audit_v1",
            "status": "GO",
            "strategy_count": len(report["strategies"]),
            "allocated_strategy_count": len(allocated),
            "allocated_weight": round(
                sum(float(row["weight"]) for row in allocated), 6
            ),
            "maximum_strategy_weight": round(
                max((float(row["weight"]) for row in allocated), default=0.0),
                6,
            ),
            "family_weights": {
                key: round(value, 6)
                for key, value in sorted(family_weights.items())
            },
            "strategy_cap_go": all(
                float(row["weight"]) <= 0.250001 for row in allocated
            ),
            "family_cap_go": all(
                value <= 0.500001 for value in family_weights.values()
            ),
            "bayesian_evidence_published": all(
                "bayesian_positive_probability" in row.get("evidence", {})
                for row in report["strategies"]
                if row.get("status") == "FROZEN_DYNAMIC"
            ),
            "missing_metrics_explicit": all(
                "missing_metrics" in row.get("evidence", {})
                for row in report["strategies"]
                if row.get("status") == "FROZEN_DYNAMIC"
            ),
            "automatic_strategy_promotion": False,
            "financial_finalist_go": False,
            "broker_calls": 0,
            "orders_generated": 0,
            "execution_authority": "NONE",
        },
        "asset_consensus.json": report["signals"],
        "portfolio_proposal.json": report["portfolio"],
        "current_signals.json": report["signals"],
        "paper_campaign.json": report["paper_campaign"],
        "live_canary_readiness.json": report["live_canary"],
        "forward_ledgers.json": {
            "schema": "dynamic_forward_ledgers_v1",
            "status": (
                "GO"
                if report["forward"]["dynamic"]["registration_count"]
                == len(
                    [
                        row
                        for row in report["strategies"]
                        if row.get("status") == "FROZEN_DYNAMIC"
                    ]
                )
                else "REGISTRATION_INCOMPLETE"
            ),
            "authority": "NONE",
            "database": "data/research/autopilot/private/research_autopilot.sqlite3",
            "forward_data_used_for_retuning": False,
            "phase11_8_registration_count": int(report["forward"].get("registration_count", 0)),
            "dynamic_registration_count": report["forward"]["dynamic"]["registration_count"],
            "dynamic_observation_count": report["forward"]["dynamic"]["observation_count"],
            "frozen_hashes": report["forward"]["dynamic"]["frozen_hashes"],
            "blockers": [],
            "strategies": [
                {
                    "strategy_id": row["strategy_id"],
                    "family": row["family"],
                    "frozen_parameters": row.get("frozen_parameters", {}),
                    "ledger_mode": "APPEND_ONLY_FORWARD_OBSERVATION",
                }
                for row in report["strategies"]
                if row.get("status") == "FROZEN_DYNAMIC"
            ],
        },
        "completion_audit.json": {
            "schema": "dynamic_multi_strategy_completion_audit_v1",
            "technical_status": "GO",
            "manual_signals_ready": report["status"]["MANUAL_SIGNALS_READY"],
            "paper_campaign_ready": report["paper_campaign"]["ready"],
            "live_canary_ready": report["live_canary"]["ready"],
            "controlled_live_ready": False,
            "blockers": sorted(
                set(
                    report["paper_campaign"]["blockers"]
                    + report["live_canary"]["blockers"]
                )
            ),
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        },
    }
    for name, payload in files.items():
        (root / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    columns = ("ticker", "action", "consensus_score", "preferred_entry", "stop_loss", "take_profit_1", "take_profit_2", "regime")
    body = "".join("<tr>" + "".join(f"<td>{row.get(col, '')}</td>" for col in columns) + "</tr>" for row in report["signals"]["signals"])
    (root / "current_signals.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Dynamic signals</title></head>"
        "<body><h1>Dynamic multi-strategy signals</h1><p>Execution authority: NONE</p><table><tr>"
        + "".join(f"<th>{col}</th>" for col in columns) + f"</tr>{body}</table></body></html>",
        encoding="utf-8",
    )
    status = report["status"]
    (root / "dynamic_engine_status.md").write_text(
        "# Dynamic Multi-Strategy Engine\n\n"
        f"Status: `{status['status']}`\n\nRegime: `{status['current_regime']}`\n\n"
        f"Strategies: {status['selected_strategy_count']}\n\n"
        f"Actionable assets: {status['actionable_asset_count']}\n\n"
        "Frozen parameters, authority NONE, no automatic execution or scaling.\n",
        encoding="utf-8",
    )
