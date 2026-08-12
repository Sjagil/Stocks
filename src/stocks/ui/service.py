from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import calendar
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import pandas as pd
import exchange_calendars as xcals

from stocks.analysis import analyze_asset, build_analysis_coverage
from stocks.market import load_market_context_map
from stocks.readiness import data_source_status
from stocks.ui.analytics import generate_statistical_artifacts


PUBLIC_ROOT = Path("output")
UNIVERSE_PATH = PUBLIC_ROOT / "universe" / "instruments.parquet"
SIGNAL_PUBLICATION = (
    PUBLIC_ROOT / "signals" / "latest_top_5_publication.json"
)
SIGNAL_SOURCE = PUBLIC_ROOT / "signals" / "latest_signals.json"
OPPORTUNITY_SOURCE = (
    PUBLIC_ROOT / "portfolio" / "opportunity_ranking.json"
)
STRATEGY_SUMMARY = (
    PUBLIC_ROOT / "research" / "phase11_14" / "strategy-summary.parquet"
)
STRATEGY_OOS_RETURNS = (
    PUBLIC_ROOT / "research" / "phase11_14" / "oos-returns.parquet"
)
STRATEGY_FORWARD_PERFORMANCE = (
    PUBLIC_ROOT / "research" / "phase11_14" / "forward-performance.json"
)
STRATEGY_FORWARD_OBSERVATION = (
    PUBLIC_ROOT
    / "research"
    / "phase11_14"
    / "latest-forward-observation.json"
)
STRATEGY_REGISTRY = (
    PUBLIC_ROOT / "research" / "strategies" / "strategy_registry.json"
)
DYNAMIC_STRATEGY_SCORES = PUBLIC_ROOT / "dynamic" / "strategy_scores.json"
DYNAMIC_STRATEGY_WEIGHTS = PUBLIC_ROOT / "dynamic" / "strategy_weights.json"
MULTITIMEFRAME_SUMMARY = (
    PUBLIC_ROOT
    / "research"
    / "phase11_10"
    / "architecture-summary.csv"
)
NEWS_DIGEST = (
    PUBLIC_ROOT / "notifications" / "market-intelligence-digest.json"
)
NEWS_INTELLIGENCE_STATUS = (
    PUBLIC_ROOT / "news" / "intelligence" / "status.json"
)
NEWS_MATERIAL_EVENTS = (
    PUBLIC_ROOT / "news" / "intelligence" / "material-events.json"
)
NEWS_PORTFOLIO_IMPACT = (
    PUBLIC_ROOT / "news" / "intelligence" / "portfolio-impact.json"
)
NEWS_EVENT_STUDY_STATUS = (
    PUBLIC_ROOT / "news" / "event_study" / "status.json"
)
GROUP_COVERAGE = PUBLIC_ROOT / "analysis" / "groups" / "coverage.json"
SECTOR_ANALYSIS = (
    PUBLIC_ROOT / "analysis" / "groups" / "sector-analysis.json"
)
INDUSTRY_ANALYSIS = (
    PUBLIC_ROOT / "analysis" / "groups" / "industry-analysis.json"
)
IBKR_NEWS_CAPABILITIES = (
    PUBLIC_ROOT / "ibkr" / "news" / "capabilities.json"
)
FORBIDDEN_KEYS = re.compile(
    r"(?:^|_)(?:account_id|account_number|raw_account|credential|"
    r"password|api_key|secret|token|fingerprint_key|private_key)(?:$|_)",
    re.IGNORECASE,
)


class ViewModelStore:
    def __init__(self, project_root: Path, ttl_seconds: float = 5.0) -> None:
        self.project_root = project_root
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, Any]] = {}
        self._frames: dict[str, tuple[int, pd.DataFrame]] = {}
        self._lock = threading.RLock()

    def dashboard(self) -> dict[str, Any]:
        return self._cached("dashboard", self._build_dashboard)

    def performance(
        self,
        month: str | None = None,
        environment: str | None = None,
    ) -> dict[str, Any]:
        selected = _validated_month(month)
        selected_environment = self._performance_environment(environment)
        return self._cached(
            f"performance:{selected}:{selected_environment}",
            lambda: self._build_performance(
                selected,
                selected_environment,
            ),
        )

    def _performance_environment(self, requested: str | None) -> str:
        normalized = str(requested or "").strip().upper()
        if normalized in {"LIVE", "PAPER"}:
            return normalized
        live = self._json(PUBLIC_ROOT / "ibkr" / "live" / "status.json")
        if str(live.get("account_reconciliation", "")).startswith(
            "LIVE_RECONCILED"
        ):
            return "LIVE"
        return "PAPER"

    def signals(
        self,
        *,
        collection: str = "diversified",
        instrument_type: str | None = None,
        sector: str | None = None,
        region: str | None = None,
        timeframe: str | None = None,
        status: str | None = None,
        minimum_score: float = 0.0,
    ) -> dict[str, Any]:
        publication = self._json(SIGNAL_PUBLICATION)
        collections = {
            "diversified": publication.get("diversified_top_5", []),
            "raw": publication.get("raw_top_5", []),
            "stocks": publication.get("top_stocks", []),
            "etfs": publication.get("top_etfs", []),
            "commodities": publication.get(
                "top_commodity_exposures", []
            ),
            "actionable": publication.get("actionable_signals", []),
            "auto": publication.get("auto_eligible_signals", []),
        }
        rows = list(collections.get(collection, collections["diversified"]))
        rows = [
            row
            for row in rows
            if (
                not instrument_type
                or instrument_type.upper()
                in str(row.get("instrument_type", "")).upper()
            )
            and (
                not sector
                or sector.casefold()
                == str(row.get("sector", "")).casefold()
            )
            and (
                not region
                or region.casefold()
                == str(row.get("region", "")).casefold()
            )
            and (
                not timeframe
                or timeframe in row.get("timeframes", [])
            )
            and (
                not status
                or status.casefold()
                == str(row.get("signal_status", "")).casefold()
            )
            and float(row.get("opportunity_score", 0.0))
            >= minimum_score
        ]
        return {
            "schema": "ui_signals_viewmodel_v1",
            "status": "GO" if publication else "NO_DATA",
            "generated_at": publication.get("generated_at"),
            "collection": collection,
            "count": len(rows),
            "signals": self._sanitize(rows),
            "trending": self._sanitize(self._trending_rows()),
            "exit_monitor": self._sanitize(self._exit_monitor()),
            "rotation_summary": self._sanitize(
                publication.get("rotation_summary", {})
            ),
            "expired_or_invalid_signal_count": publication.get(
                "expired_or_invalid_signal_count", 0
            ),
            "available_collections": sorted(collections),
            "execution_authority": "NONE",
            "automatic_submission": False,
        }

    def _trending_rows(self) -> list[dict[str, Any]]:
        ranking = self._json(OPPORTUNITY_SOURCE)
        source = self._json(SIGNAL_SOURCE)
        publication = self._json(SIGNAL_PUBLICATION)
        target_allocation = self._json(
            PUBLIC_ROOT / "portfolio" / "target_allocation.json"
        )
        dynamic_risk = self._json(
            PUBLIC_ROOT / "portfolio" / "dynamic-risk-state.json"
        )
        published = {
            str(row.get("symbol", "")).upper(): row
            for row in publication.get("raw_top_5", [])
            if row.get("symbol")
        }
        targets = {
            str(row.get("ticker", "")).upper(): row
            for row in target_allocation.get("allocations", [])
            if row.get("ticker")
        }
        now = datetime.now(UTC)
        best_signals: dict[str, dict[str, Any]] = {}
        latest_references: dict[str, dict[str, Any]] = {}
        for row in source.get("signals", []):
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("ticker") or row.get("asset") or "").upper()
            if symbol and row.get("market_reference_price") is not None:
                previous_reference = latest_references.get(symbol)
                fetched = _timestamp(row.get("market_reference_fetched_at"))
                previous_fetched = _timestamp(
                    previous_reference.get("market_reference_fetched_at")
                    if previous_reference
                    else None
                )
                if previous_reference is None or (
                    fetched is not None
                    and (
                        previous_fetched is None or fetched > previous_fetched
                    )
                ):
                    latest_references[symbol] = row
            if row.get("data_freshness") != "FRESH":
                continue
            expiration = _timestamp(row.get("expiration_timestamp"))
            if expiration is None or expiration < now:
                continue
            if not symbol:
                continue
            previous = best_signals.get(symbol)
            if previous is None or (
                (_finite(row.get("confidence_score")) or 0.0)
                > (_finite(previous.get("confidence_score")) or 0.0)
            ):
                best_signals[symbol] = row

        result = []
        for rank, opportunity in enumerate(ranking.get("opportunities", [])[:20], 1):
            symbol = str(opportunity.get("ticker", "")).upper()
            signal = best_signals.get(symbol, {})
            reference = latest_references.get(symbol, {})
            published_signal = published.get(symbol, {})
            target = targets.get(symbol, {})
            score = _finite(opportunity.get("opportunity_score")) or 0.0
            eligible = bool(opportunity.get("research_allocation_eligible"))
            resolved = bool(opportunity.get("contract_resolved"))
            actionable = (
                str(published_signal.get("signal_status", "")).upper()
                == "ACTIONABLE"
            )
            allocated = bool(target)
            if eligible and resolved and actionable and signal and allocated:
                model_action = "BUY_SETUP"
            elif eligible and resolved and actionable and signal:
                model_action = "WAITLIST_CAPITAL_CONSTRAINED"
            else:
                model_action = "WATCH"
            operational_slots = int(
                dynamic_risk.get("operational_maximum_positions", 0) or 0
            )
            draft_order = None
            if model_action == "BUY_SETUP" and signal:
                draft_order = {
                    "side": "BUY",
                    "order_type": "LIMIT",
                    "quantity": (
                        signal.get("suggested_quantity")
                        if operational_slots > 0
                        else None
                    ),
                    "quantity_source": (
                        "INDICATIVE_SIGNAL_QUANTITY"
                        if operational_slots > 0
                        else "BLOCKED_CAPITAL_LEVEL_ZERO"
                    ),
                    "limit_price": signal.get("limit_entry_price"),
                    "protective_stop": signal.get("stop_loss"),
                    "target_1": signal.get("take_profit_1"),
                    "target_2": signal.get("take_profit_2"),
                    "status": (
                        "DRAFT_NOT_SUBMITTABLE_AUTHORITY_NONE"
                        if operational_slots > 0
                        else "DRAFT_BLOCKED_CAPITAL_LEVEL_ZERO"
                    ),
                }
            expiration = _timestamp(signal.get("expiration_timestamp"))
            expiry_hours = (
                max(0.0, (expiration - now).total_seconds() / 3600.0)
                if expiration is not None
                else None
            )
            base_risk = _finite(dynamic_risk.get("base_risk_per_trade")) or 0.0
            risk_multiplier = _finite(
                dynamic_risk.get("multipliers", {}).get("combined")
            ) or 0.0
            risk_of_equity = min(
                _finite(dynamic_risk.get("maximum_risk_per_trade")) or 0.0,
                base_risk * risk_multiplier * score,
            )
            components = opportunity.get("components", {})
            failure_reasons = list(signal.get("risks", []))
            failure_reasons.extend(opportunity.get("deployment_blockers", []))
            reference_validity = str(
                reference.get("price_validity_status") or ""
            ).strip()
            if reference_validity and reference_validity != "CURRENT_ENTRY_REFERENCE_GO":
                failure_reasons.append(reference_validity)
            if opportunity.get("event_risk", 0):
                failure_reasons.append("EVENT_RISK_PRESENT")
            result.append(
                {
                    "rank": rank,
                    "symbol": symbol,
                    "heat": "HOT" if score >= 0.75 else "TRENDING" if score >= 0.65 else "WATCH",
                    "model_action": model_action,
                    "signal_status": published_signal.get(
                        "signal_status", "WATCH"
                    ),
                    "signal_action": signal.get("action", "NO_SIGNAL"),
                    "classification": _signal_classification(score),
                    "opportunity_score": score,
                    "confidence": _finite(signal.get("confidence_score")),
                    "timeframe": "/".join(opportunity.get("timeframes", [])) or signal.get("timeframe"),
                    "current_price": signal.get("current_market_price")
                    or reference.get("current_market_price")
                    or reference.get("market_reference_price"),
                    "price_validity_status": signal.get(
                        "price_validity_status"
                    )
                    or reference.get("price_validity_status"),
                    "entry_instruction": signal.get("entry_instruction")
                    or reference.get("entry_instruction"),
                    "market_reference_age_minutes": signal.get(
                        "market_reference_age_minutes"
                    )
                    or reference.get("market_reference_age_minutes"),
                    "entry_low": signal.get("entry_zone_low"),
                    "entry_high": signal.get("entry_zone_high"),
                    "stop": signal.get("stop_loss") or signal.get("invalidation_level"),
                    "target_1": signal.get("take_profit_1"),
                    "target_2": signal.get("take_profit_2"),
                    "reward_risk_1": signal.get("reward_risk_1"),
                    "expected_rr": _finite(signal.get("reward_risk_1")),
                    "recommended_weight": _finite(
                        target.get("research_target_weight")
                    ),
                    "risk_of_equity": risk_of_equity,
                    "macro_fit": _finite(components.get("regime_fit")),
                    "regime": signal.get("regime"),
                    "regime_confidence": _finite(
                        dynamic_risk.get("regime_confidence")
                    ),
                    "factor_cluster": (
                        f"{opportunity.get('sector', 'UNKNOWN')} / "
                        f"{opportunity.get('region', 'UNKNOWN')}"
                    ),
                    "correlation_penalty": _finite(
                        target.get("correlation_penalty")
                    ),
                    "strategy_id": signal.get("strategy_id"),
                    "strategy_families": opportunity.get(
                        "strategy_families", []
                    ),
                    "why_it_exists": list(signal.get("reasons", []))[:8],
                    "why_it_can_fail": sorted(set(failure_reasons))[:10],
                    "sector": opportunity.get("sector"),
                    "region": opportunity.get("region"),
                    "freshness": signal.get("data_freshness", "NO_CURRENT_SIGNAL"),
                    "expiration": signal.get("expiration_timestamp"),
                    "expiry_hours": (
                        round(expiry_hours, 2)
                        if expiry_hours is not None
                        else None
                    ),
                    "draft_order": draft_order,
                    "execution_authority": "NONE",
                }
            )
        return result

    def _exit_monitor(self) -> dict[str, Any]:
        lifecycle = self._json(
            PUBLIC_ROOT / "operations" / "signal-lifecycle.json"
        )
        machine = self._json(
            PUBLIC_ROOT / "operations" / "machine-status.json"
        )
        rows = lifecycle.get("rows", [])
        exits = []
        lifecycle_avoids = []
        for row in rows:
            status = str(row.get("lifecycle_status", "")).upper()
            compact = {
                "symbol": row.get("ticker"),
                "strategy_id": row.get("strategy_id"),
                "previous_action": row.get("previous_action"),
                "current_action": row.get("current_action"),
                "lifecycle_status": status,
                "model_action": (
                    "SELL_EXIT" if status == "EXIT" else "AVOID_NEW_LONG"
                ),
                "sell_order_status": (
                    "BLOCKED_POSITION_IDENTITY_NOT_RECONCILED"
                    if status == "EXIT"
                    else "NOT_AN_ORDER"
                ),
                "reason_codes": [],
                "source": "SIGNAL_LIFECYCLE",
            }
            if status == "EXIT":
                exits.append(compact)
            elif status == "AVOID":
                lifecycle_avoids.append(compact)
        latest_signals = self._json(SIGNAL_SOURCE)
        invalidated_by_symbol: dict[str, dict[str, Any]] = {}
        for row in latest_signals.get("signals", []):
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("ticker") or row.get("asset") or "").upper()
            validity = str(row.get("price_validity_status") or "").upper()
            invalidated = (
                str(row.get("lifecycle_status") or "").upper()
                == "INVALIDATED"
                or str(row.get("action") or "").upper() == "AVOID"
                or (
                    validity
                    and validity != "CURRENT_ENTRY_REFERENCE_GO"
                )
            )
            if not symbol or not invalidated:
                continue
            previous = invalidated_by_symbol.get(symbol)
            fetched = _timestamp(row.get("market_reference_fetched_at"))
            previous_fetched = _timestamp(
                previous.get("market_reference_fetched_at")
                if previous
                else None
            )
            if previous is None or (
                fetched is not None
                and (previous_fetched is None or fetched > previous_fetched)
            ):
                invalidated_by_symbol[symbol] = row
        ranking = self._json(OPPORTUNITY_SOURCE)
        rank_by_symbol = {
            str(row.get("ticker", "")).upper(): index
            for index, row in enumerate(
                ranking.get("opportunities", []), start=1
            )
            if isinstance(row, dict) and row.get("ticker")
        }
        market_avoids = []
        for symbol, row in sorted(
            invalidated_by_symbol.items(),
            key=lambda item: (
                rank_by_symbol.get(item[0], 1_000_000),
                item[0],
            ),
        ):
            reason_codes = list(row.get("risks", []))
            validity = str(row.get("price_validity_status") or "").strip()
            if validity and validity not in reason_codes:
                reason_codes.append(validity)
            market_avoids.append(
                {
                    "symbol": symbol,
                    "strategy_id": row.get("strategy_id"),
                    "previous_action": row.get("original_action"),
                    "current_action": row.get("action"),
                    "lifecycle_status": "INVALIDATED",
                    "model_action": "AVOID_NEW_LONG",
                    "sell_order_status": "NOT_AN_ORDER",
                    "reason_codes": reason_codes,
                    "current_price": row.get("current_market_price")
                    or row.get("market_reference_price"),
                    "source": "MARKET_REFERENCE",
                }
            )
        avoids = []
        avoided_symbols: set[str] = set()
        for row in [*market_avoids, *lifecycle_avoids]:
            symbol = str(row.get("symbol", "")).upper()
            if not symbol or symbol in avoided_symbols:
                continue
            avoids.append(row)
            avoided_symbols.add(symbol)
            if len(avoids) >= 20:
                break
        position_management = self._json(
            PUBLIC_ROOT / "portfolio" / "position-management.json"
        )
        existing = {
            (str(row.get("symbol", "")).upper(), row.get("model_action"))
            for row in exits
        }
        action_map = {
            "EXIT": "SELL_EXIT",
            "REDUCE": "SELL_REDUCE",
            "REDUCE_50": "SELL_REDUCE",
            "TAKE_PARTIAL_PROFIT": "SELL_PARTIAL",
            "TAKE_PARTIAL_25": "SELL_PARTIAL",
            "TAKE_PARTIAL_50": "SELL_PARTIAL",
            "UPDATE_TRAILING_STOP": "PROTECTIVE_STOP_UPDATE",
        }
        for row in position_management.get("positions", []):
            advisory = str(row.get("advisory_action", "")).upper()
            model_action = action_map.get(advisory)
            symbol = str(row.get("ticker", "")).upper()
            if not model_action or not symbol or (symbol, model_action) in existing:
                continue
            exits.append(
                {
                    "symbol": symbol,
                    "strategy_id": "PORTFOLIO_POSITION_MANAGER",
                    "previous_action": None,
                    "current_action": advisory,
                    "lifecycle_status": advisory,
                    "model_action": model_action,
                    "sell_order_status": "ADVISORY_AUTHORITY_NONE",
                    "reason_codes": list(row.get("reason_codes", [])),
                    "current_r": row.get("current_r"),
                    "peak_r": row.get("peak_r"),
                    "profit_giveback": row.get("profit_giveback"),
                    "market_data_status": row.get("market_data_status"),
                    "market_data_reason": row.get("market_data_reason"),
                    "market_data_age_minutes": row.get(
                        "market_data_age_minutes"
                    ),
                    "source": "POSITION_MANAGEMENT",
                }
            )
            existing.add((symbol, model_action))
        return {
            "status": lifecycle.get("status", "NO_DATA"),
            "generated_at": lifecycle.get("generated_at"),
            "exit_signals": exits,
            "exit_signal_count": len(exits),
            "avoid_signals": avoids,
            "avoid_signal_count": len(avoids),
            "observed_position_count": machine.get("last_open_positions", 0),
            "observed_open_order_count": machine.get("last_open_orders", 0),
            "sell_orders_submittable": False,
            "position_management_status": position_management.get(
                "status", "UNAVAILABLE"
            ),
            "execution_authority": "NONE",
        }

    def universe(
        self,
        *,
        query: str = "",
        instrument_type: str | None = None,
        sector: str | None = None,
        industry: str | None = None,
        region: str | None = None,
        eligibility: str | None = None,
        page: int = 1,
        page_size: int = 50,
        sort: str = "symbol",
    ) -> dict[str, Any]:
        frame = self._frame(UNIVERSE_PATH)
        if frame.empty:
            return {
                "schema": "ui_universe_viewmodel_v1",
                "status": "NO_DATA",
                "count": 0,
                "instruments": [],
            }
        work = frame
        if query:
            needle = re.escape(query.strip())
            work = work.loc[
                work["symbol"].astype(str).str.contains(
                    needle, case=False, regex=True
                )
                | work["name"].astype(str).str.contains(
                    needle, case=False, regex=True
                )
            ]
        for column, value in (
            ("instrument_type", instrument_type),
            ("sector", sector),
            ("industry", industry),
            ("region", region),
            ("eligibility_status", eligibility),
        ):
            if value:
                work = work.loc[
                    work[column].astype(str).str.casefold()
                    == value.casefold()
                ]
        allowed_sort = {
            "symbol",
            "instrument_type",
            "sector",
            "industry",
            "region",
            "eligibility_status",
        }
        sort = sort if sort in allowed_sort else "symbol"
        work = work.sort_values([sort, "symbol"], kind="stable")
        page = max(1, int(page))
        page_size = min(100, max(10, int(page_size)))
        total = len(work)
        start = (page - 1) * page_size
        columns = [
            "symbol",
            "name",
            "instrument_type",
            "asset_type",
            "exposure_type",
            "sector",
            "industry",
            "region",
            "country",
            "active_listing",
            "signal_eligible",
            "live_executable",
            "eligibility_status",
            "compliance_status",
        ]
        return {
            "schema": "ui_universe_viewmodel_v1",
            "status": "GO",
            "count": total,
            "page": page,
            "page_size": page_size,
            "page_count": max(1, (total + page_size - 1) // page_size),
            "instruments": self._sanitize(
                work.iloc[start : start + page_size][columns].to_dict(
                    orient="records"
                )
            ),
            "filters": {
                "instrument_types": self._values(
                    frame, "instrument_type"
                ),
                "sectors": self._values(frame, "sector"),
                "industries": self._values(frame, "industry"),
                "regions": self._values(frame, "region"),
                "eligibility": self._values(
                    frame, "eligibility_status"
                ),
            },
            "execution_authority": "NONE",
        }

    def dimension(self, name: str) -> dict[str, Any]:
        allowed = {"sector", "industry", "region"}
        if name not in allowed:
            return {"status": "BLOCKED", "reason": "UNKNOWN_DIMENSION"}
        payload = self._json(
            PUBLIC_ROOT / "universe" / f"{name}-ranking.json"
        )
        if name == "region":
            exchange_clock = _exchange_clock()
            payload["exchange_clock"] = exchange_clock
            payload["exchange_clock_generated_at"] = datetime.now(
                UTC
            ).isoformat()
            payload["exchange_open_count"] = sum(
                row["status"] == "OPEN" for row in exchange_clock
            )
            payload["exchange_closed_count"] = sum(
                row["status"] == "CLOSED" for row in exchange_clock
            )
        return self._sanitize(payload)

    def instruments_by_type(self, instrument_type: str) -> dict[str, Any]:
        result = self.universe(
            instrument_type=instrument_type,
            page_size=100,
        )
        collection = (
            "commodities"
            if instrument_type == "COMMODITY_EXPOSURE"
            else "etfs"
        )
        signal_view = self.signals(collection=collection)
        result["signals"] = signal_view.get("signals", [])
        result["signal_count"] = signal_view.get("count", 0)
        return result

    def strategies(self) -> dict[str, Any]:
        return self._cached("strategies", self._build_strategies)

    def portfolio(self) -> dict[str, Any]:
        return self._cached("portfolio", self._build_portfolio)

    def research(self) -> dict[str, Any]:
        return self._cached("research", self._build_research)

    def health(self) -> dict[str, Any]:
        return self._cached("health", self._build_health, ttl=15.0)

    def audit(self) -> dict[str, Any]:
        return self._cached("audit", self._build_audit)

    def asset(self, symbol: str) -> dict[str, Any]:
        return self._sanitize(analyze_asset(self.project_root, symbol))

    def analysis_coverage(self) -> dict[str, Any]:
        return self._cached(
            "analysis_coverage",
            lambda: build_analysis_coverage(
                self.project_root,
                publish=True,
            ),
            ttl=300.0,
        )

    def news(self) -> dict[str, Any]:
        return self._cached("news", self._build_news, ttl=60.0)

    def chart(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 300,
    ) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        if not re.fullmatch(r"[A-Z0-9.^=-]{1,20}", symbol):
            return {"status": "BLOCKED", "reason": "INVALID_SYMBOL"}
        if interval not in {"1h", "2h", "4h", "1d", "1w", "1mo"}:
            return {"status": "BLOCKED", "reason": "INVALID_INTERVAL"}
        candidates = list(
            (
                self.project_root
                / "data"
                / "research"
                / "multitimeframe"
                / "private"
            ).glob(
                f"provider=*/symbol={symbol}/interval={interval}/"
                "source_interval=*/bars.parquet"
            )
        )
        if not candidates:
            return {
                "schema": "ui_chart_v1",
                "status": "NO_DATA",
                "symbol": symbol,
                "interval": interval,
                "bars": [],
            }
        preferred = sorted(
            candidates,
            key=lambda path: (
                "provider=EODHD" not in str(path),
                "provider=YFINANCE" not in str(path),
                -path.stat().st_mtime_ns,
            ),
        )[0]
        frame = pd.read_parquet(preferred)
        frame = frame.loc[
            ~frame.get("is_partial", pd.Series(False, index=frame.index))
            .fillna(False)
            .astype(bool)
        ]
        frame = frame.tail(min(1000, max(50, limit))).copy()
        timestamp = pd.to_datetime(
            frame["timestamp_utc"], utc=True, errors="coerce"
        )
        bars = []
        for index, row in frame.iterrows():
            ts = timestamp.loc[index]
            if pd.isna(ts):
                continue
            bars.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": _finite(row.get("open")),
                    "high": _finite(row.get("high")),
                    "low": _finite(row.get("low")),
                    "close": _finite(row.get("close")),
                    "volume": _finite(row.get("volume")),
                }
            )
        return {
            "schema": "ui_chart_v1",
            "status": "GO" if bars else "NO_DATA",
            "symbol": symbol,
            "interval": interval,
            "provider": str(frame["provider"].iloc[-1])
            if not frame.empty and "provider" in frame
            else "LOCAL_CACHE",
            "bar_origin": str(frame["bar_origin"].iloc[-1])
            if not frame.empty and "bar_origin" in frame
            else "UNKNOWN",
            "bars": bars,
        }

    def event_fingerprint(self) -> str:
        paths = (
            Path("runtime/heartbeat.json"),
            SIGNAL_PUBLICATION,
            PUBLIC_ROOT / "operations" / "machine-status.json",
            PUBLIC_ROOT / "ibkr" / "live" / "status.json",
            PUBLIC_ROOT / "ibkr" / "live" / "reconciliation.json",
            PUBLIC_ROOT / "portfolio" / "status.json",
            PUBLIC_ROOT / "research" / "multitimeframe" / "status.json",
            PUBLIC_ROOT / "market_context" / "status.json",
            PUBLIC_ROOT / "ibkr" / "data-capabilities" / "capability-matrix.json",
            PUBLIC_ROOT / "ibkr" / "live" / "writer-integrity-verify.json",
            PUBLIC_ROOT / "research" / "sec_intelligence" / "status.json",
        )
        state = []
        for relative in paths:
            path = self.project_root / relative
            state.append(
                (
                    str(relative),
                    path.stat().st_mtime_ns if path.is_file() else 0,
                    path.stat().st_size if path.is_file() else 0,
                )
            )
        return hashlib.sha256(
            json.dumps(state, sort_keys=True).encode()
        ).hexdigest()

    def _build_dashboard(self) -> dict[str, Any]:
        heartbeat = self._json(Path("runtime/heartbeat.json"))
        machine = self._json(
            PUBLIC_ROOT / "operations" / "machine-status.json"
        )
        execution = self._json(
            PUBLIC_ROOT / "operations" / "execution-status.json"
        )
        live = self._json(PUBLIC_ROOT / "ibkr" / "live" / "status.json")
        live_reconciliation = self._json(
            PUBLIC_ROOT / "ibkr" / "live" / "reconciliation.json"
        )
        writer_integrity = self._json(
            PUBLIC_ROOT
            / "ibkr"
            / "live"
            / "writer-integrity-verify.json"
        )
        phase9 = self._json(
            PUBLIC_ROOT / "ibkr" / "phase9" / "status.json"
        )
        data_capabilities = self._json(
            PUBLIC_ROOT
            / "ibkr"
            / "data-capabilities"
            / "capability-matrix.json"
        )
        sec_intelligence = self._json(
            PUBLIC_ROOT
            / "research"
            / "sec_intelligence"
            / "status.json"
        )
        current_reconciliation = str(
            live_reconciliation.get("reconciliation_status") or ""
        )
        has_current_reconciliation = bool(current_reconciliation)
        effective_reconciliation = (
            current_reconciliation
            if has_current_reconciliation
            else str(live.get("account_reconciliation") or "UNKNOWN")
        )
        effective_broker_status = (
            str(live_reconciliation.get("status") or "NO_GO")
            if has_current_reconciliation
            else str(heartbeat.get("IBKR_status") or "UNKNOWN")
        )
        reconciliation_go = (
            effective_broker_status == "GO"
            and effective_reconciliation.startswith("LIVE_RECONCILED")
        )
        live_blockers = {
            str(blocker)
            for blocker in live.get("open_blockers", [])
            if blocker
        }
        if has_current_reconciliation and not reconciliation_go:
            live_blockers.update(
                str(blocker)
                for blocker in live_reconciliation.get("blockers", [])
                if blocker
            )
            live_blockers.add(effective_reconciliation)
        portfolio = self._json(PUBLIC_ROOT / "portfolio" / "status.json")
        exposure = self._json(
            PUBLIC_ROOT / "portfolio" / "exposure_report.json"
        )
        portfolio_risk = self._json(
            PUBLIC_ROOT / "portfolio" / "risk_contributions.json"
        )
        dynamic_risk = self._json(
            PUBLIC_ROOT / "portfolio" / "dynamic-risk-state.json"
        )
        dynamic = self._json(PUBLIC_ROOT / "dynamic" / "status.json")
        capital = self._json(
            PUBLIC_ROOT / "capital" / "current_level.json"
        )
        daily_target = self._json(
            PUBLIC_ROOT / "capital" / "daily_profit_target.json"
        )
        signals = self.signals(collection="diversified")
        coverage = self.analysis_coverage()
        news = self.news()
        account = self._latest_private_account_metrics()
        performance = self._build_performance(
            datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m"),
            self._performance_environment(None),
        )
        period_performance = {
            str(row.get("period")): row
            for row in performance.get("period_performance", [])
            if isinstance(row, dict) and row.get("period")
        }
        macro = self._macro_dashboard()
        data_sources = data_source_status(self.project_root)
        data_readiness = self._json(
            PUBLIC_ROOT / "reports" / "data_readiness.json"
        )
        market_context_status = self._json(
            PUBLIC_ROOT / "market_context" / "status.json"
        )
        asset_context_status = self._json(
            PUBLIC_ROOT / "market_context" / "asset-context.json"
        )
        entry_observer_status = self._json(
            PUBLIC_ROOT / "market_context" / "entry-observer-status.json"
        )
        episode_outcome_status = self._json(
            PUBLIC_ROOT
            / "market_context"
            / "entry-episode-completeness.json"
        )
        role_leaderboards = self._json(
            PUBLIC_ROOT
            / "research"
            / "role_leaderboards"
            / "status.json"
        )
        active_swing_sprints = self._json(
            PUBLIC_ROOT / "research" / "active_swing" / "status.json"
        )
        active_swing_coverage = self._json(
            PUBLIC_ROOT
            / "research"
            / "active_swing"
            / "shortlist_data"
            / "coverage.json"
        )
        entry_filter_experiment = self._json(
            PUBLIC_ROOT
            / "research"
            / "active_swing"
            / "entry_filter_experiment"
            / "results.json"
        )
        active_swing_leaderboards = self._json(
            PUBLIC_ROOT
            / "research"
            / "active_swing"
            / "leaderboards"
            / "status.json"
        )
        selective_ml = self._json(
            PUBLIC_ROOT
            / "research"
            / "active_swing"
            / "selective_ml"
            / "status.json"
        )
        rl_status = self._json(PUBLIC_ROOT / "rl" / "status.json")
        p4_readiness = self._json(
            PUBLIC_ROOT / "verification" / "p4-readiness.json"
        )
        rejected_shadow = self._json(
            PUBLIC_ROOT
            / "research"
            / "active_swing"
            / "rejected_shadow"
            / "status.json"
        )
        gate_attribution = self._json(
            PUBLIC_ROOT
            / "research"
            / "active_swing"
            / "rejected_shadow"
            / "gate-attribution.json"
        )
        evidence_throughput = self._json(
            PUBLIC_ROOT
            / "research"
            / "evidence_throughput"
            / "status.json"
        )
        market_context = load_market_context_map(self.project_root)
        flow_leaders = sorted(
            (
                {
                    "symbol": symbol,
                    "score": context.get("orderflow", {}).get(
                        "ranking_score", 0.5
                    ),
                    "raw_score": context.get("orderflow", {}).get(
                        "raw_score", 0.0
                    ),
                    "confidence": context.get("orderflow", {}).get(
                        "confidence", 0.0
                    ),
                    "status": context.get("orderflow", {}).get("status"),
                }
                for symbol, context in market_context.items()
            ),
            key=lambda row: (-float(row["score"]), row["symbol"]),
        )
        return self._sanitize(
            {
                "schema": "ui_dashboard_viewmodel_v1",
                "status": "GO",
                "generated_at": datetime.now(UTC).isoformat(),
                "runtime": {
                    "status": heartbeat.get(
                        "runtime_status", machine.get("status", "UNKNOWN")
                    ),
                    "state": heartbeat.get("runtime_state", "UNKNOWN"),
                    "mode": machine.get("mode", "UNKNOWN"),
                    "cycle_count": machine.get("cycle_count", 0),
                    "last_heartbeat": heartbeat.get("last_heartbeat"),
                },
                "broker": {
                    "ibkr_status": effective_broker_status,
                    "live_connection": (
                        effective_reconciliation
                        if has_current_reconciliation
                        else live.get("broker_connection", "UNKNOWN")
                    ),
                    "reconciliation": effective_reconciliation,
                    "reconciliation_go": reconciliation_go,
                    "status_source": (
                        "CURRENT_RECONCILIATION_ARTIFACT"
                        if has_current_reconciliation
                        else "COMPOSITE_LIVE_STATUS_FALLBACK"
                    ),
                    "paper_positions": heartbeat.get("open_positions"),
                    "paper_orders": heartbeat.get("open_orders"),
                    "phase9_status": phase9.get("status", "UNAVAILABLE"),
                    "phase9_open_blockers": phase9.get("open_blockers", []),
                    "market_data_type": next(
                        (
                            row.get("market_data_type")
                            for row in data_capabilities.get("rows", [])
                            if row.get("market_data_type") is not None
                        ),
                        None,
                    ),
                    "data_capability_status": data_capabilities.get(
                        "status", "UNAVAILABLE"
                    ),
                    "data_capabilities": data_capabilities.get(
                        "summary", {}
                    ),
                    "missing_subscription_classes": data_capabilities.get(
                        "missing_subscription_classes", []
                    ),
                },
                "authority": {
                    "execution": execution.get(
                        "execution_authority", "NONE"
                    ),
                    "strategy": execution.get(
                        "strategy_authority", "NONE"
                    ),
                    "live": live.get("execution_authority", "NONE"),
                    "automatic_submission": False,
                    "writer_hash_integrity": writer_integrity.get(
                        "writer_hash_integrity", False
                    ),
                    "writer_integrity_status": writer_integrity.get(
                        "status", "UNAVAILABLE"
                    ),
                    "live_trading_allowed": False,
                },
                "risk": {
                    "kill_switch": (
                        "TRIGGERED_RECONCILIATION"
                        if has_current_reconciliation
                        and not reconciliation_go
                        else live.get("kill_switch_state", "UNKNOWN")
                    ),
                    "capital_level": capital.get(
                        "CURRENT_CAPITAL_LEVEL_NAME", "UNKNOWN"
                    ),
                    "target_exposure": capital.get(
                        "CURRENT_TARGET_EXPOSURE", 0.0
                    ),
                    "research_gross_exposure": exposure.get(
                        "research_gross_exposure", 0.0
                    ),
                    "approved_gross_exposure": exposure.get(
                        "approved_gross_exposure", 0.0
                    ),
                    "drawdown": live.get("drawdown", "UNAVAILABLE"),
                    "portfolio_drawdown": dynamic_risk.get(
                        "portfolio_drawdown_pct"
                    ),
                    "portfolio_drawdown_status": dynamic_risk.get(
                        "portfolio_drawdown_status", "UNAVAILABLE"
                    ),
                    "drawdown_velocity": dynamic_risk.get(
                        "drawdown_velocity_per_day"
                    ),
                    "drawdown_velocity_status": dynamic_risk.get(
                        "drawdown_velocity_status", "UNAVAILABLE"
                    ),
                    "equity_band": dynamic_risk.get(
                        "equity_band", "UNAVAILABLE"
                    ),
                    "dynamic_maximum_positions": dynamic_risk.get(
                        "dynamic_research_maximum_positions", 0
                    ),
                    "operational_maximum_positions": dynamic_risk.get(
                        "operational_maximum_positions", 0
                    ),
                    "portfolio_heat": portfolio_risk.get(
                        "research_portfolio_heat", 0.0
                    ),
                    "open_risk": portfolio_risk.get(
                        "observed_current_portfolio_heat"
                    ),
                    "open_risk_status": (
                        "GO"
                        if portfolio_risk.get(
                            "observed_current_portfolio_heat"
                        )
                        is not None
                        else "UNAVAILABLE"
                    ),
                    "cash_target": exposure.get(
                        "research_cash_weight", 1.0
                    ),
                    "combined_risk_multiplier": dynamic_risk.get(
                        "multipliers", {}
                    ).get("combined", 0.0),
                    "daily_pnl": live.get("daily_pnl_eur", "UNAVAILABLE"),
                    "daily_profit_target_status": daily_target.get(
                        "status", "UNAVAILABLE"
                    ),
                    "daily_profit_target_pct": daily_target.get(
                        "target_pct"
                    ),
                },
                "account": account,
                "performance": {
                    period: period_performance.get(
                        period,
                        {
                            "period": period,
                            "status": "NO_OBSERVATIONS",
                            "net_pnl_eur": None,
                            "return_pct": None,
                        },
                    )
                    for period in (
                        "TODAY",
                        "WEEK_TO_DATE",
                        "MONTH_TO_DATE",
                        "YEAR_TO_DATE",
                    )
                },
                "market": {
                    "regime": dynamic.get(
                        "current_regime", "UNAVAILABLE"
                    ),
                    "regime_confidence": dynamic_risk.get(
                        "regime_confidence"
                    ),
                    "regime_confidence_status": (
                        "GO"
                        if dynamic_risk.get("regime_confidence")
                        is not None
                        else "UNAVAILABLE"
                    ),
                    "market_state": heartbeat.get(
                        "market_state", "UNKNOWN"
                    ),
                    "observed_timeframes": portfolio.get(
                        "observed_timeframes", []
                    ),
                    "news_freshness": news.get(
                        "freshness_status", "UNAVAILABLE"
                    ),
                    "event_risk_within_24h": news.get(
                        "event_risk_within_24h", False
                    ),
                    "intraday_data_status": data_sources.get(
                        "multitimeframe_current_data_status",
                        "CURRENT_DATA_STATUS_UNAVAILABLE",
                    ),
                    "intraday_data_ratio": data_sources.get(
                        "current_data_ratio"
                    ),
                    "fresh_intraday_pair_count": data_sources.get(
                        "fresh_current_symbol_interval_pairs", 0
                    ),
                    "requested_intraday_pair_count": data_sources.get(
                        "requested_current_symbol_interval_pairs", 0
                    ),
                    "core_data_readiness": data_readiness.get(
                        "status", "UNAVAILABLE"
                    ),
                    "open_data_gap_count": len(
                        data_readiness.get("open_data_gaps", [])
                    ),
                    "microstructure_data_status": data_readiness.get(
                        "layers", {}
                    ).get("observed_equity_microstructure", {}).get(
                        "status", "UNAVAILABLE"
                    ),
                },
                "macro": macro,
                "market_context": {
                    "status": market_context_status.get(
                        "context_readiness", "UNAVAILABLE"
                    ),
                    "observed_at": market_context_status.get("observed_at"),
                    "symbol_count": len(market_context),
                    "gex_symbol_count": market_context_status.get(
                        "gex", {}
                    ).get("available_symbol_count", 0),
                    "observed_equity_orderflow": market_context_status.get(
                        "orderflow", {}
                    ).get("observed_trade_flow_available", False),
                    "bar_flow_proxy_only": not market_context_status.get(
                        "orderflow", {}
                    ).get("observed_trade_flow_available", False),
                    "flow_leaders": flow_leaders[:5],
                    "asset_context_count": asset_context_status.get(
                        "context_count", 0
                    ),
                    "entry_observer_status": entry_observer_status.get(
                        "status", "UNAVAILABLE"
                    ),
                    "entry_shortlist_count": entry_observer_status.get(
                        "shortlist_count", 0
                    ),
                    "entry_state_counts": entry_observer_status.get(
                        "state_counts", {}
                    ),
                    "entry_signal_funnel": entry_observer_status.get(
                        "signal_funnel", {}
                    ),
                    "entry_asset_profile_counts": (
                        entry_observer_status.get(
                            "asset_profile_counts", {}
                        )
                    ),
                    "entry_ml_status": entry_observer_status.get(
                        "ml_status", "NOT_TRAINED"
                    ),
                    "episode_completion_status": episode_outcome_status.get(
                        "status", "NOT_RUN"
                    ),
                    "episode_completion_ratio": episode_outcome_status.get(
                        "completion_ratio", 0.0
                    ),
                    "terminal_episode_count": episode_outcome_status.get(
                        "terminal_episode_count", 0
                    ),
                    "pending_episode_count": episode_outcome_status.get(
                        "pending_episode_count", 0
                    ),
                    "entry_market_regime": entry_observer_status.get(
                        "market_regime", "UNAVAILABLE"
                    ),
                    "bar_proxy_can_confirm_entry": entry_observer_status.get(
                        "bar_proxy_can_confirm_entry", False
                    ),
                    "authority": "CONTEXT_ONLY",
                    "execution_authority": "NONE",
                },
                "research": {
                    "signal_count": portfolio.get("signal_count", 0),
                    "opportunity_count": portfolio.get(
                        "opportunity_count", 0
                    ),
                    "strategy_dna_count": portfolio.get(
                        "registered_strategy_dna_count", 0
                    ),
                    "financial_finalist": False,
                    "role_leaderboards": role_leaderboards.get(
                        "roles", {}
                    ),
                    "active_swing_sprints": active_swing_sprints.get(
                        "components", {}
                    ),
                    "active_swing_funnel": active_swing_coverage.get(
                        "funnel", {}
                    ),
                    "observed_tape_count": active_swing_coverage.get(
                        "observed_tape_count", 0
                    ),
                    "observed_depth_count": active_swing_coverage.get(
                        "observed_depth_count", 0
                    ),
                    "entry_filter_closed_episode_count": (
                        entry_filter_experiment.get(
                            "closed_independent_base_episodes", 0
                        )
                    ),
                    "entry_filter_promotion_eligible": (
                        entry_filter_experiment.get(
                            "promotion_eligible", False
                        )
                    ),
                    "active_swing_champions": {
                        role: value.get("rows", [])[:1]
                        for role, value in active_swing_leaderboards.get(
                            "roles", {}
                        ).items()
                        if isinstance(value, dict)
                    },
                    "selective_ml_status": selective_ml.get(
                        "status", "NOT_RUN"
                    ),
                    "selective_ml_closed_labels": selective_ml.get(
                        "closed_trainable_episode_count", 0
                    ),
                    "selective_ml_label_source": selective_ml.get(
                        "canonical_label_source",
                        "PHASE9_CANONICAL_BROKER_FILL",
                    ),
                    "selective_ml_shadow_only": (
                        selective_ml.get("model_authority", "NONE")
                        == "NONE"
                    ),
                    "ml_regime_conditioning": selective_ml.get(
                        "regime_conditioning", "NOT_IMPLEMENTED"
                    ),
                    "ml_regime_dataset_status": selective_ml.get(
                        "regime_dataset", {}
                    ).get("status", "NOT_EVALUABLE"),
                    "ml_regime_count": selective_ml.get(
                        "regime_dataset", {}
                    ).get("available_regime_count", 0),
                    "ml_loro_status": selective_ml.get(
                        "regime_generalization", {}
                    ).get("status", "NOT_EVALUABLE"),
                    "ml_loro_evaluable_regime_count": selective_ml.get(
                        "regime_generalization", {}
                    ).get("evaluable_regime_count", 0),
                    "ml_worst_regime": selective_ml.get(
                        "regime_generalization", {}
                    ).get("worst_regime"),
                    "ml_worst_regime_auc": selective_ml.get(
                        "regime_generalization", {}
                    ).get("worst_regime_auc"),
                    "ml_regime_auc_std": selective_ml.get(
                        "regime_generalization", {}
                    ).get("regime_auc_std"),
                    "ml_robust_generalization_score": selective_ml.get(
                        "regime_generalization", {}
                    ).get("robust_generalization_score"),
                    "ml_reinforcement_learning_status": selective_ml.get(
                        "reinforcement_learning_status",
                        "DISABLED_PREMATURE_SAMPLE_SIZE",
                    )
                    if not rl_status
                    else rl_status.get("status", "NOT_RUN"),
                    "rl_status": rl_status.get("status", "NOT_RUN"),
                    "rl_mode": rl_status.get("rl_mode", "DISABLED"),
                    "rl_active_policy": rl_status.get("active_policy"),
                    "rl_challenger_policy": rl_status.get(
                        "challenger_policy"
                    ),
                    "rl_last_inference": rl_status.get("last_inference"),
                    "rl_episode_count": rl_status.get("episodes", 0),
                    "rl_closed_episode_count": rl_status.get(
                        "closed_episodes", 0
                    ),
                    "rl_mean_reward": rl_status.get("mean_reward"),
                    "rl_rolling_reward": rl_status.get("rolling_reward"),
                    "rl_net_pnl": rl_status.get("net_pnl"),
                    "rl_maximum_drawdown": rl_status.get(
                        "maximum_drawdown"
                    ),
                    "rl_policy_entropy": rl_status.get("policy_entropy"),
                    "rl_action_distribution": rl_status.get(
                        "action_distribution", {}
                    ),
                    "rl_trade_frequency": rl_status.get(
                        "trade_frequency", 0.0
                    ),
                    "rl_skip_frequency": rl_status.get(
                        "skip_frequency", 0.0
                    ),
                    "rl_promotion_status": rl_status.get(
                        "promotion_status", "NOT_ELIGIBLE"
                    ),
                    "rl_next_evaluation": rl_status.get("next_evaluation"),
                    "rl_next_training_check": rl_status.get(
                        "next_training_check"
                    ),
                    "rl_training_status": rl_status.get(
                        "training_status", {}
                    ),
                    "rl_alerts": rl_status.get("alerts", []),
                    "rl_reward_by_regime": rl_status.get(
                        "reward_by_regime", {}
                    ),
                    "p4_status": p4_readiness.get(
                        "status", "NOT_PUBLISHED"
                    ),
                    "p4_complete": p4_readiness.get("p4_complete", False),
                    "p4_external_gates": p4_readiness.get(
                        "economic_and_external_gates", {}
                    ),
                    "p4_external_blockers": p4_readiness.get(
                        "economic_and_external_blockers", []
                    ),
                    "rejected_shadow_status": rejected_shadow.get(
                        "status", "NOT_RUN"
                    ),
                    "rejected_episode_count": rejected_shadow.get(
                        "rejected_episode_count", 0
                    ),
                    "counterfactual_performance_count": (
                        rejected_shadow.get(
                            "counterfactual_performance_count", 0
                        )
                    ),
                    "gate_attribution_status": gate_attribution.get(
                        "status", "NOT_RUN"
                    ),
                    "gate_sample_ready_count": rejected_shadow.get(
                        "gate_sample_ready_count", 0
                    ),
                    "automatic_gate_relaxation": gate_attribution.get(
                        "automatic_gate_relaxation", False
                    ),
                    "evidence_throughput_status": (
                        evidence_throughput.get("status", "NOT_RUN")
                    ),
                    "evidence_funnel": evidence_throughput.get(
                        "funnel", {}
                    ),
                    "validation_throughput": evidence_throughput.get(
                        "validation", {}
                    ),
                    "near_finalist_count": evidence_throughput.get(
                        "finalists", {}
                    ).get("near_finalist_count", 0),
                    "closest_finalist_candidate": (
                        evidence_throughput.get("finalists", {}).get(
                            "closest_candidate"
                        )
                    ),
                    "sec_overlay_status": sec_intelligence.get(
                        "status", "UNAVAILABLE"
                    ),
                    "sec_structured_event_count": sec_intelligence.get(
                        "structured_event_count", 0
                    ),
                    "sec_metadata_event_count": sec_intelligence.get(
                        "metadata_event_count", 0
                    ),
                    "sec_max_overlay_points": sec_intelligence.get(
                        "max_overlay_points", 4.0
                    ),
                    "sec_authority": sec_intelligence.get(
                        "authority", "RANKING_OVERLAY_ONLY"
                    ),
                    "sec_standalone_entry_allowed": False,
                    "global_cross_role_ranking_allowed": (
                        role_leaderboards.get(
                            "global_cross_role_ranking_allowed", False
                        )
                    ),
                    "analyzable_instrument_count": coverage.get(
                        "analyzable_instrument_count", 0
                    ),
                    "one_hour_instrument_count": coverage.get(
                        "one_hour_instrument_count", 0
                    ),
                    "two_hour_instrument_count": coverage.get(
                        "two_hour_instrument_count", 0
                    ),
                },
                "top_signals": signals.get("signals", []),
                "latest_news": news.get("important_news", [])[:5],
                "blockers": sorted(live_blockers),
            }
        )

    def _macro_dashboard(self) -> dict[str, Any]:
        score = self._json(PUBLIC_ROOT / "macro" / "score.json")
        history = self._json(PUBLIC_ROOT / "macro" / "history.json")
        features = score.get("features", {})
        composites = score.get("scores", {})

        def unavailable(
            label: str,
            series_id: str,
            reason: str = "SERIES_UNAVAILABLE",
        ) -> dict[str, Any]:
            return {
                "label": label,
                "series_id": series_id,
                "value": None,
                "display_value": "Unavailable",
                "status": "UNAVAILABLE",
                "reason": reason,
                "observed_at": None,
                "provider": None,
                "direction": "UNKNOWN",
            }

        def feature(
            label: str,
            series_id: str,
            *,
            unit: str,
            decimals: int = 2,
        ) -> dict[str, Any]:
            row = features.get(series_id)
            if not isinstance(row, dict):
                return unavailable(label, series_id)
            value = _finite(row.get("original_value"))
            transformed = _finite(row.get("transformed_value"))
            status = str(row.get("status") or "UNAVAILABLE").upper()
            if value is None:
                return {
                    **unavailable(
                        label,
                        series_id,
                        str(row.get("reason") or "VALUE_UNAVAILABLE"),
                    ),
                    "status": status,
                    "observed_at": row.get("observation_date"),
                    "provider": row.get("provider"),
                }
            suffix = {
                "percent": "%",
                "percentage_points": " pp",
                "index": "",
                "ratio": "",
            }.get(unit, "")
            direction = (
                "RISING"
                if transformed is not None and transformed > 0
                else "FALLING"
                if transformed is not None and transformed < 0
                else "FLAT_OR_UNKNOWN"
            )
            return {
                "label": label,
                "series_id": series_id,
                "value": value,
                "display_value": f"{value:,.{decimals}f}{suffix}",
                "unit": unit,
                "status": status,
                "reason": row.get("reason"),
                "observed_at": row.get("observation_date"),
                "provider": row.get("provider"),
                "direction": direction,
                "age_days": _finite(row.get("age_days")),
                "quality_status": row.get("quality_status"),
            }

        def composite(label: str, score_id: str) -> dict[str, Any]:
            row = composites.get(score_id)
            if not isinstance(row, dict):
                return unavailable(label, score_id)
            value = _finite(row.get("value"))
            status = str(row.get("status") or "UNAVAILABLE").upper()
            return {
                "label": label,
                "series_id": score_id,
                "value": value,
                "display_value": (
                    f"{value:+.1f}" if value is not None else "Unavailable"
                ),
                "unit": "normalized_score",
                "status": status,
                "reason": (
                    None
                    if value is not None
                    else "COMPOSITE_VALUE_UNAVAILABLE"
                ),
                "observed_at": score.get("as_of"),
                "provider": "MACRO_COMPOSITE",
                "direction": (
                    "POSITIVE"
                    if value is not None and value > 0
                    else "NEGATIVE"
                    if value is not None and value < 0
                    else "NEUTRAL_OR_UNKNOWN"
                ),
                "confidence": _finite(row.get("confidence")),
                "coverage": _finite(row.get("coverage")),
                "missing_inputs": list(row.get("missing_inputs", [])),
            }

        indicators = [
            feature("DXY", "USD_INDEX", unit="index"),
            feature(
                "Real rate (US 10Y)",
                "US_REAL_YIELD_10Y",
                unit="percent",
            ),
            feature(
                "Yields (10Y-2Y curve)",
                "US_YIELD_CURVE_10Y2Y",
                unit="percentage_points",
            ),
            feature(
                "Credit spread (US HY)",
                "US_HIGH_YIELD_SPREAD",
                unit="percentage_points",
            ),
            feature("VIX", "VIX", unit="index"),
            composite("Liquidity", "liquidity"),
            composite("Commodities", "commodity"),
            feature(
                "Global breadth",
                "EQUITY_BREADTH_GLOBAL",
                unit="ratio",
                decimals=3,
            ),
            feature(
                "EM versus developed",
                "EM_DEVELOPED_RELATIVE_STRENGTH",
                unit="ratio",
                decimals=3,
            ),
        ]

        current_regime = score.get("regime", {})
        current_name = str(
            current_regime.get("overall_macro_regime") or "UNAVAILABLE"
        )
        previous_name = None
        records = history.get("history", [])
        for record in reversed(records if isinstance(records, list) else []):
            if not isinstance(record, dict):
                continue
            candidate = record.get("regime", {})
            name = candidate.get("overall_macro_regime")
            if name:
                previous_name = str(name)
                break
        regime_change = (
            "UNAVAILABLE"
            if current_name == "UNAVAILABLE" or previous_name is None
            else "REGIME_CHANGED"
            if current_name != previous_name
            else "REGIME_STABLE"
        )
        data_quality = score.get("data_quality", {})
        quality_status = str(
            data_quality.get("status") or "UNAVAILABLE"
        ).upper()
        status = (
            "NO_DATA"
            if not score
            else "GO"
            if quality_status == "GO"
            else "DEGRADED_DATA_INCOMPLETE"
        )
        return {
            "schema": "ui_macro_dashboard_v1",
            "status": status,
            "as_of": score.get("as_of"),
            "data_quality_status": quality_status,
            "feature_status_counts": data_quality.get(
                "feature_status_counts", {}
            ),
            "indicators": indicators,
            "current_regime": current_name,
            "previous_regime": previous_name,
            "regime_change": regime_change,
            "hysteresis_status": current_regime.get(
                "hysteresis_status", "UNAVAILABLE"
            ),
            "regime_confidence": _finite(
                current_regime.get("confidence")
            ),
            "regime_reasons": list(current_regime.get("reasons", [])),
            "macro_analysis_authority": score.get(
                "macro_analysis_authority", "RESEARCH_ONLY"
            ),
            "execution_authority": "NONE",
            "predictive_claim": False,
        }

    def _build_strategies(self) -> dict[str, Any]:
        summary = self._frame(STRATEGY_SUMMARY)
        oos_returns = self._frame(STRATEGY_OOS_RETURNS)
        multitimeframe = self._frame(MULTITIMEFRAME_SUMMARY)
        registry = self._json(STRATEGY_REGISTRY)
        dynamic_scores = self._json(DYNAMIC_STRATEGY_SCORES)
        dynamic_weights = self._json(DYNAMIC_STRATEGY_WEIGHTS)
        forward_performance = self._json(STRATEGY_FORWARD_PERFORMANCE)
        forward_observation = self._json(STRATEGY_FORWARD_OBSERVATION)
        observer_rows = []
        for row in forward_observation.get("observations", []):
            raw_signals = row.get("raw_active_signals", [])
            target_weights = row.get(
                "current_attested_target_weights", {}
            )
            observer_rows.append(
                {
                    "strategy_id": row.get("strategy_id"),
                    "formula": row.get("formula"),
                    "timeframe": row.get("timeframe"),
                    "asset_class": row.get("asset_class"),
                    "observer_tier": row.get(
                        "observer_tier",
                        "ROBUST_FORWARD_OBSERVER",
                    ),
                    "portfolio_eligible": bool(
                        row.get("portfolio_eligible", False)
                    ),
                    "execution_eligible": False,
                    "freshness": row.get("data_freshness"),
                    "closed_bar_timestamp": row.get(
                        "closed_bar_timestamp"
                    ),
                    "active_signal_count": len(raw_signals),
                    "active_symbols": [
                        str(signal.get("symbol"))
                        for signal in raw_signals
                        if signal.get("symbol")
                    ],
                    "attested_target_count": len(target_weights),
                    "portfolio_action": row.get("portfolio_action"),
                }
            )
        observer_rows.sort(
            key=lambda row: (
                row["observer_tier"]
                != "ROBUST_FORWARD_OBSERVER",
                str(row["timeframe"]),
                str(row["strategy_id"]),
            )
        )
        score_map = {
            str(row.get("strategy_id")): row
            for row in dynamic_scores.get("strategies", [])
            if row.get("strategy_id")
        }
        dynamic_rows = []
        for weight in dynamic_weights.get("weights", []):
            if float(weight.get("weight", 0.0) or 0.0) <= 0:
                continue
            score = score_map.get(str(weight.get("strategy_id")), {})
            evidence = score.get("evidence", {})
            bayesian = evidence.get("bayesian_positive_probability", {})
            dynamic_rows.append(
                {
                    "strategy_id": weight.get("strategy_id"),
                    "family": weight.get("family"),
                    "timeframe": weight.get("timeframe"),
                    "score": _finite(weight.get("score")),
                    "weight": _finite(weight.get("weight")),
                    "evidence_status": weight.get("evidence_status"),
                    "bayesian_probability": _finite(
                        bayesian.get("probability_above_break_even")
                    ),
                    "sample_count": evidence.get("sample_count", 0),
                    "metric_coverage": _finite(
                        evidence.get("metric_coverage")
                    ),
                    "missing_metrics": evidence.get("missing_metrics", []),
                    "financial_finalist": bool(
                        score.get("financial_finalist", False)
                    ),
                }
            )
        dynamic_rows.sort(
            key=lambda row: (
                -(row["weight"] or 0.0),
                str(row["strategy_id"]),
            )
        )
        rows = []
        if not summary.empty:
            columns = [
                "strategy_id",
                "formula",
                "asset_class",
                "timeframe",
                "combined_oos_CAGR",
                "combined_oos_Sharpe",
                "combined_period_profit_factor",
                "maximum_drawdown",
                "positive_fold_ratio",
                "cost_50bps_combined_return",
                "research_pass",
                "robust_pass",
                "deployable_pass",
                "deployment_blockers",
                "financial_finalist",
            ]
            available = [column for column in columns if column in summary]
            rows = (
                summary.sort_values(
                    ["robust_pass", "combined_oos_Sharpe"],
                    ascending=False,
                )[available]
                .head(50)
                .to_dict(orient="records")
            )
        timeframe_summary = []
        multitimeframe_rows = []
        if not multitimeframe.empty:
            for timeframe in ("1h", "2h", "4h", "1d", "1w"):
                subset = multitimeframe.loc[
                    multitimeframe["lower_timeframe"].eq(timeframe)
                ]
                qualified = subset.loc[
                    subset["median_oos_CAGR"].gt(0)
                    & subset["median_oos_portfolio_pf"].gt(1)
                    & subset["median_fill_count"].ge(30)
                ]
                stress = qualified.loc[
                    qualified["cost_50bps_median_pf"].gt(1)
                ]
                best = (
                    qualified.sort_values(
                        [
                            "cost_50bps_median_pf",
                            "median_oos_Sharpe",
                        ],
                        ascending=False,
                    ).head(1)
                )
                timeframe_summary.append(
                    {
                        "timeframe": timeframe,
                        "tested_architectures": len(subset),
                        "positive_pf_architectures": len(qualified),
                        "stress_survivors": len(stress),
                        "best_architecture": (
                            str(best.iloc[0]["architecture"])
                            if not best.empty
                            else "NONE"
                        ),
                        "best_stress_pf": (
                            _finite(
                                best.iloc[0]["cost_50bps_median_pf"]
                            )
                            if not best.empty
                            else None
                        ),
                    }
                )
            candidates = multitimeframe.loc[
                multitimeframe["median_oos_CAGR"].gt(0)
                & multitimeframe["median_oos_portfolio_pf"].gt(1)
                & multitimeframe["median_fill_count"].ge(30)
            ].copy()
            candidates["research_tier"] = "RESEARCH_LEAD"
            candidates.loc[
                candidates["cost_50bps_median_pf"].gt(1)
                & candidates["positive_fold_ratio"].ge(0.55),
                "research_tier",
            ] = "COST_STRESS_SURVIVOR"
            candidate_columns = [
                "architecture",
                "entry_strategy",
                "higher_timeframe",
                "middle_timeframe",
                "lower_timeframe",
                "fold_count",
                "median_oos_CAGR",
                "median_oos_Sharpe",
                "median_oos_portfolio_pf",
                "cost_50bps_median_pf",
                "worst_oos_drawdown",
                "positive_fold_ratio",
                "median_fill_count",
                "research_tier",
            ]
            multitimeframe_rows = (
                candidates.sort_values(
                    [
                        "cost_50bps_median_pf",
                        "median_oos_Sharpe",
                    ],
                    ascending=False,
                )[candidate_columns]
                .head(50)
                .replace({float("nan"): None})
                .to_dict(orient="records")
            )
        analytics = self._cached(
            "statistical_artifacts",
            lambda: generate_statistical_artifacts(self.project_root),
            ttl=300.0,
        )
        monitoring = self._strategy_monitoring(
            dynamic_scores=dynamic_scores,
            dynamic_weights=dynamic_weights,
            summary=summary,
            oos_returns=oos_returns,
            forward_performance=forward_performance,
        )
        return self._sanitize(
            {
                "schema": "ui_strategies_viewmodel_v2",
                "status": "GO" if registry or rows else "NO_DATA",
                "registered_strategy_count": registry.get(
                    "bulk_strategy_count", 0
                ),
                "standard_strategy_count": registry.get(
                    "standard_strategy_count", 0
                ),
                "validated_strategy_count": len(summary),
                "research_pass_count": int(
                    summary.get("research_pass", pd.Series(dtype=bool)).sum()
                ),
                "robust_pass_count": int(
                    summary.get("robust_pass", pd.Series(dtype=bool)).sum()
                ),
                "financial_finalist_count": int(
                    summary.get(
                        "financial_finalist", pd.Series(dtype=bool)
                    ).sum()
                ),
                "strategies": rows,
                "dynamic_strategy_count": len(dynamic_rows),
                "dynamic_allocated_weight": dynamic_weights.get(
                    "allocated_weight", 0.0
                ),
                "dynamic_unallocated_weight": dynamic_weights.get(
                    "unallocated_weight", 1.0
                ),
                "dynamic_strategies": dynamic_rows,
                "strategy_monitoring": monitoring,
                "forward_observers": observer_rows,
                "forward_observer_count": len(observer_rows),
                "exploratory_forward_observer_count": sum(
                    row["observer_tier"]
                    == "EXPLORATORY_FORWARD_OBSERVER"
                    for row in observer_rows
                ),
                "forward_observer_active_signal_count": sum(
                    int(row["active_signal_count"])
                    for row in observer_rows
                ),
                "multitimeframe_architecture_count": len(
                    multitimeframe
                ),
                "timeframe_summary": timeframe_summary,
                "multitimeframe_candidates": multitimeframe_rows,
                "statistical_artifacts": analytics,
                "strategy_authority": "NONE",
                "execution_authority": "NONE",
            }
        )

    def _strategy_monitoring(
        self,
        *,
        dynamic_scores: dict[str, Any],
        dynamic_weights: dict[str, Any],
        summary: pd.DataFrame,
        oos_returns: pd.DataFrame,
        forward_performance: dict[str, Any],
    ) -> dict[str, Any]:
        summary_map = {
            str(row["strategy_id"]): row
            for row in summary.to_dict(orient="records")
            if row.get("strategy_id")
        }
        weight_map = {
            str(row.get("strategy_id")): row
            for row in dynamic_weights.get("weights", [])
            if row.get("strategy_id")
        }
        rolling_map = _rolling_strategy_performance(oos_returns)
        forward_map = {
            str(row.get("strategy_id")): row
            for row in forward_performance.get("per_strategy", [])
            if row.get("strategy_id")
        }
        rows = []
        lifecycle_counts = {
            "ACTIVE": 0,
            "PAPER": 0,
            "SHADOW": 0,
            "PAUSED": 0,
            "RESEARCH": 0,
        }
        for score in dynamic_scores.get("strategies", []):
            strategy_id = str(score.get("strategy_id") or "")
            if not strategy_id:
                continue
            enabled = bool(score.get("enabled", False))
            classification = str(
                score.get("classification") or "RESEARCH_ONLY"
            ).upper()
            if not enabled:
                lifecycle = "PAUSED"
            elif classification in {"CONTROLLED_LIVE", "PORTFOLIO_ELIGIBLE"}:
                lifecycle = "ACTIVE"
            elif "PAPER" in classification:
                lifecycle = "PAPER"
            elif classification == "FROZEN_SHADOW":
                lifecycle = "SHADOW"
            else:
                lifecycle = "RESEARCH"
            lifecycle_counts[lifecycle] += 1

            evidence = score.get("evidence", {})
            metrics = evidence.get("metrics", {})
            summary_row = summary_map.get(strategy_id, {})
            rolling = rolling_map.get(
                strategy_id,
                {
                    "status": "UNAVAILABLE_NO_MATCHING_OOS_RETURNS",
                    "window_observations": 0,
                    "available_observations": 0,
                    "return": None,
                    "period_profit_factor": None,
                    "annualized_sharpe": None,
                    "maximum_drawdown": None,
                    "last_observation": None,
                    "cost_bps": None,
                    "freshness_status": "UNAVAILABLE",
                    "age_days": None,
                },
            )
            weight = weight_map.get(strategy_id, {})
            forward = forward_map.get(
                strategy_id,
                {
                    "strategy_id": strategy_id,
                    "independent_session_count": 0,
                    "episode_count": 0,
                    "closed_episode_count": 0,
                    "open_episode_count": 0,
                    "net_profit_factor": None,
                    "profit_factor_reason": "NO_FORWARD_OBSERVATION",
                    "net_expectancy": None,
                    "win_rate": None,
                    "sample_status": "INSUFFICIENT_SAMPLE",
                },
            )
            profit_factor = _metric_raw(metrics, "profit_factor")
            if profit_factor is None:
                profit_factor = _finite(
                    summary_row.get("combined_period_profit_factor")
                )
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "family": score.get("family") or weight.get("family"),
                    "timeframe": score.get("timeframe")
                    or weight.get("timeframe"),
                    "classification": classification,
                    "lifecycle": lifecycle,
                    "enabled": enabled,
                    "active": lifecycle == "ACTIVE",
                    "paper": lifecycle == "PAPER",
                    "shadow": lifecycle == "SHADOW",
                    "paused": lifecycle == "PAUSED",
                    "score": _finite(score.get("score")),
                    "weight": _finite(weight.get("weight")),
                    "profit_factor": profit_factor,
                    "expectancy": _metric_raw(metrics, "expectancy"),
                    "maximum_drawdown": _finite(
                        summary_row.get("maximum_drawdown")
                    ),
                    "regime_fit": _metric_raw(metrics, "regime_fit"),
                    "regime_fit_status": metrics.get(
                        "regime_fit", {}
                    ).get("status", "UNAVAILABLE"),
                    "sample_count": evidence.get("sample_count", 0),
                    "evidence_status": evidence.get(
                        "evidence_status", "UNAVAILABLE"
                    ),
                    "rolling": rolling,
                    "forward": forward,
                    "financial_finalist": bool(
                        score.get("financial_finalist", False)
                    ),
                    "deployment_eligible": bool(
                        score.get("deployment_eligible", False)
                    ),
                    "strategy_authority": "NONE",
                    "execution_authority": "NONE",
                }
            )
        order = {"ACTIVE": 0, "PAPER": 1, "SHADOW": 2, "RESEARCH": 3, "PAUSED": 4}
        rows.sort(
            key=lambda row: (
                order.get(str(row["lifecycle"]), 9),
                -(row["weight"] or 0.0),
                str(row["strategy_id"]),
            )
        )
        return {
            "schema": "ui_strategy_monitoring_v1",
            "status": "GO" if rows else "NO_DATA",
            "current_regime": self._json(
                PUBLIC_ROOT / "dynamic" / "status.json"
            ).get("current_regime", "UNAVAILABLE"),
            "counts": lifecycle_counts,
            "strategy_count": len(rows),
            "rows": rows,
            "rolling_metric_type": "OOS_PERIOD_RETURNS_NOT_TRADE_PNL",
            "rolling_cost_layer_bps": 10.0,
            "forward_metric_type": (
                "INDEPENDENT_POINT_IN_TIME_CLOSED_EPISODE_RETURNS"
            ),
            "forward_status": forward_performance.get(
                "status", "NOT_AVAILABLE"
            ),
            "forward_evidence_end": forward_performance.get("evidence_end"),
            "forward_counts": forward_performance.get("counts", {}),
            "forward_aggregate": forward_performance.get("aggregate", {}),
            "forward_cost_layer_bps_per_side": forward_performance.get(
                "cost_model", {}
            ).get("cost_bps_per_side"),
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
        }

    def _build_portfolio(self) -> dict[str, Any]:
        names = {
            "status": PUBLIC_ROOT / "portfolio" / "status.json",
            "current_allocation": PUBLIC_ROOT
            / "portfolio"
            / "current_allocation.json",
            "target_allocation": PUBLIC_ROOT
            / "portfolio"
            / "target_allocation.json",
            "exposures": PUBLIC_ROOT
            / "portfolio"
            / "exposure_report.json",
            "risk_contributions": PUBLIC_ROOT
            / "portfolio"
            / "risk_contributions.json",
            "dynamic_risk": PUBLIC_ROOT
            / "portfolio"
            / "dynamic-risk-state.json",
            "position_management": PUBLIC_ROOT
            / "portfolio"
            / "position-management.json",
            "confluence": PUBLIC_ROOT
            / "portfolio"
            / "confluence-audit.json",
            "rebalance_plan": PUBLIC_ROOT
            / "portfolio"
            / "rebalance_plan.json",
            "coverage_waterfall": PUBLIC_ROOT
            / "portfolio"
            / "coverage-waterfall.json",
            "normalized_opportunities": PUBLIC_ROOT
            / "portfolio"
            / "normalized-opportunities.json",
            "overlap": PUBLIC_ROOT
            / "portfolio"
            / "overlap-report.json",
            "desired_targets": PUBLIC_ROOT
            / "portfolio"
            / "desired-portfolio-targets.json",
            "monitoring": PUBLIC_ROOT
            / "portfolio"
            / "p1-monitoring.json",
            "cross_asset_intelligence": PUBLIC_ROOT
            / "portfolio"
            / "cross-asset-intelligence.json",
            "p1_readiness": PUBLIC_ROOT
            / "portfolio"
            / "p1-readiness.json",
            "performance_attribution": PUBLIC_ROOT
            / "portfolio"
            / "performance-attribution.json",
            "capital": PUBLIC_ROOT / "capital" / "current_level.json",
            "daily_profit_target": PUBLIC_ROOT
            / "capital"
            / "daily_profit_target.json",
        }
        analytics = self._cached(
            "statistical_artifacts",
            lambda: generate_statistical_artifacts(self.project_root),
            ttl=300.0,
        )
        machine = self._json(PUBLIC_ROOT / "operations" / "machine-status.json")
        phase9 = self._json(PUBLIC_ROOT / "ibkr" / "phase9" / "status.json")
        live = self._json(PUBLIC_ROOT / "ibkr" / "live" / "status.json")
        current_live_reconciliation = self._json(
            PUBLIC_ROOT / "ibkr" / "live" / "reconciliation.json"
        )
        current = self._json(names["current_allocation"])
        explicit_reconciliation = str(
            current_live_reconciliation.get("reconciliation_status") or ""
        )
        if explicit_reconciliation:
            live_reconciliation = (
                current_live_reconciliation.get("status") == "GO"
                and explicit_reconciliation.startswith("LIVE_RECONCILED")
            )
        else:
            live_reconciliation = str(
                live.get("account_reconciliation", "")
            ).startswith("LIVE_RECONCILED")
        paper_reconciliation = bool(
            phase9.get("checks", {}).get("reconciliation")
        )
        if live_reconciliation:
            environment = "LIVE_READ_ONLY"
            observed_positions = int(live.get("position_count") or 0)
            observed_orders = int(live.get("open_order_count") or 0)
        elif paper_reconciliation:
            environment = "PAPER_READ_ONLY"
            observed_positions = int(machine.get("last_open_positions") or 0)
            observed_orders = int(machine.get("last_open_orders") or 0)
        else:
            environment = "BROKER_OBSERVATION_BLOCKED"
            observed_positions = None
            observed_orders = None
        mirror_positions = int(current.get("position_count") or 0)
        broker_state = {
            "environment": environment,
            "status": (
                "BROKER_OBSERVATION_BLOCKED"
                if observed_positions is None
                else "BROKER_MIRROR_CONFLICT_BLOCKED"
                if observed_positions != mirror_positions
                else "BROKER_COUNTS_ALIGNED"
            ),
            "observed_position_count": observed_positions,
            "observed_open_order_count": observed_orders,
            "mirror_position_count": mirror_positions,
            "reconciliation": (
                "GO" if live_reconciliation or paper_reconciliation else "NO_GO"
            ),
            "reconciliation_status": (
                explicit_reconciliation
                or live.get("account_reconciliation")
                or "UNAVAILABLE"
            ),
            "last_observed_at": (
                live.get("last_heartbeat")
                if live_reconciliation
                else machine.get("last_account_reconciliation")
            ),
            "paper_observed_position_count": int(
                machine.get("last_open_positions") or 0
            ),
            "paper_observed_open_order_count": int(
                machine.get("last_open_orders") or 0
            ),
            "execution_authority": "NONE",
        }
        handover_checks = [
            {"label": "Broker reconciliation", "go": broker_state["reconciliation"] == "GO"},
            {
                "label": "Position mirror aligned",
                "go": observed_positions is not None
                and observed_positions == mirror_positions,
            },
            {
                "label": "No open broker orders",
                "go": observed_orders is not None and observed_orders == 0,
            },
            {
                "label": "Whole-share sizing",
                "go": self._json(names["status"]).get(
                    "private_whole_share_sizing_status"
                )
                in {"GO", "GO_WITH_CONSTRAINTS"},
            },
            {"label": "Execution authority", "go": False},
        ]
        proactive_queue = []
        for row in self._trending_rows():
            if row["model_action"] != "BUY_SETUP":
                continue
            proactive_queue.append(
                {
                    "symbol": row["symbol"],
                    "action": "OPEN_LONG_RESEARCH_CANDIDATE",
                    "score": row["opportunity_score"],
                    "target_weight": next(
                        (
                            item.get("research_target_weight")
                            for item in self._json(names["target_allocation"]).get(
                                "allocations", []
                            )
                            if item.get("ticker") == row["symbol"]
                        ),
                        None,
                    ),
                    "draft_order": row["draft_order"],
                    "status": "ADVISORY_AUTHORITY_NONE",
                }
            )
        position_overview = self._broker_position_overview(
            current=current,
            position_management=self._json(names["position_management"]),
            broker_state=broker_state,
        )
        normalized = self._json(names["normalized_opportunities"])
        combined = normalized.get("combined_ranking", [])
        opportunity_intelligence = {
            "top_by_asset_class": {
                asset_class: [
                    row
                    for row in combined
                    if row.get("asset_class") == asset_class
                ][:10]
                for asset_class in (
                    "EQUITY",
                    "ETF",
                    "COMMODITY_EXPOSURE",
                    "CASH",
                )
            },
            "combined_ranking": combined[:25],
            "research_opportunity_is_execution_eligible": False,
            "execution_authority": "NONE",
        }
        return self._sanitize(
            {
                "schema": "ui_portfolio_viewmodel_v1",
                "status": "GO",
                **{
                    name: self._json(path)
                    for name, path in names.items()
                },
                "statistical_artifacts": analytics,
                "broker_state": broker_state,
                "position_overview": position_overview,
                "handover_checks": handover_checks,
                "handover_ready": all(row["go"] for row in handover_checks),
                "proactive_queue": proactive_queue,
                "opportunity_intelligence": opportunity_intelligence,
                "execution_authority": "NONE",
            }
        )

    def _broker_position_overview(
        self,
        *,
        current: dict[str, Any],
        position_management: dict[str, Any],
        broker_state: dict[str, Any],
    ) -> dict[str, Any]:
        public_positions = {
            str(row.get("position_identity", "")): row
            for row in current.get("positions", [])
            if isinstance(row, dict) and row.get("position_identity")
        }
        if not public_positions:
            return {
                "status": (
                    "EMPTY_COMPLETE"
                    if current.get("status")
                    == "PRIVATE_BROKER_POSITION_SNAPSHOT_COMPLETE"
                    else current.get("status", "UNAVAILABLE")
                ),
                "position_count": 0,
                "positions": [],
                "financial_values_scope": "LOCAL_READ_ONLY_UI_ONLY",
                "strategy_ownership_inferred": False,
                "execution_authority": "NONE",
            }

        management = {
            str(row.get("position_identity", "")): row
            for row in position_management.get("positions", [])
            if isinstance(row, dict) and row.get("position_identity")
        }
        private_states = self._private_position_management_states()
        sizing = self._verified_private_position_sizing()
        sizing_by_symbol = {
            str(row.get("ticker", "")).upper(): row
            for row in sizing.get("positions", [])
            if isinstance(row, dict) and row.get("ticker")
        }
        universe = self._frame(UNIVERSE_PATH)
        universe_by_symbol = (
            {
                str(row.get("symbol", "")).upper(): row
                for row in universe.to_dict(orient="records")
                if row.get("symbol")
            }
            if not universe.empty
            else {}
        )
        ranking = self._json(OPPORTUNITY_SOURCE)
        context_by_symbol = {
            str(row.get("ticker", "")).upper(): row
            for row in ranking.get("opportunities", [])
            if isinstance(row, dict) and row.get("ticker")
        }
        equity = _finite(sizing.get("account_equity_eur"))
        rows = []
        for identity, public in sorted(
            public_positions.items(),
            key=lambda item: str(item[1].get("ticker", "")),
        ):
            symbol = str(public.get("ticker", "")).upper()
            lifecycle = management.get(identity, {})
            private = private_states.get(identity, {})
            private_sizing = sizing_by_symbol.get(symbol, {})
            instrument = universe_by_symbol.get(symbol, {})
            context = context_by_symbol.get(symbol, {})
            current_quantity = _finite(
                private.get("quantity")
                if private
                else private_sizing.get("current_quantity")
            )
            unit_notional = _finite(private_sizing.get("unit_notional_eur"))
            risk_per_share = _finite(private_sizing.get("risk_per_share_eur"))
            position_weight = (
                current_quantity * unit_notional / equity
                if current_quantity is not None
                and unit_notional is not None
                and equity is not None
                and equity > 0
                else None
            )
            risk_contribution = (
                current_quantity * risk_per_share / equity
                if current_quantity is not None
                and risk_per_share is not None
                and equity is not None
                and equity > 0
                else None
            )
            sector = str(
                context.get("sector")
                or instrument.get("sector")
                or "UNKNOWN_SECTOR"
            )
            region = str(
                context.get("region")
                or instrument.get("region")
                or "UNKNOWN_REGION"
            )
            private_available = bool(private)
            rows.append(
                {
                    "symbol": symbol,
                    "position_identity": identity,
                    "asset_class": (
                        instrument.get("instrument_type")
                        or instrument.get("asset_type")
                        or public.get("security_type")
                        or "UNKNOWN"
                    ),
                    "currency": public.get("currency", "UNKNOWN"),
                    "strategy": "UNATTRIBUTED_BROKER_POSITION",
                    "strategy_context": context.get(
                        "strategy_families", []
                    ),
                    "entry_price": (
                        _finite(private.get("entry_price"))
                        if private_available
                        else None
                    ),
                    "entry_price_source": (
                        "BROKER_AVERAGE_COST"
                        if private_available
                        else "PRIVATE_POSITION_STATE_UNAVAILABLE"
                    ),
                    "current_price": (
                        _finite(private.get("current_price"))
                        if private_available
                        else None
                    ),
                    "current_price_source": private.get(
                        "market_data_source"
                    ),
                    "stop": (
                        _finite(private.get("proposed_stop"))
                        or _finite(private.get("initial_stop"))
                        if private_available
                        else None
                    ),
                    "stop_source": (
                        "ADVISORY_POSITION_MANAGER"
                        if private_available
                        else "PRIVATE_POSITION_STATE_UNAVAILABLE"
                    ),
                    "quantity": current_quantity,
                    "current_r": lifecycle.get(
                        "current_r", private.get("current_r")
                    ),
                    "peak_r": lifecycle.get(
                        "peak_r", private.get("peak_r")
                    ),
                    "profit_giveback": lifecycle.get(
                        "profit_giveback",
                        private.get("profit_giveback"),
                    ),
                    "position_weight": (
                        round(position_weight, 8)
                        if position_weight is not None
                        else None
                    ),
                    "risk_contribution": (
                        round(risk_contribution, 8)
                        if risk_contribution is not None
                        else None
                    ),
                    "factor_cluster": f"{sector} / {region}",
                    "status": lifecycle.get(
                        "status", private.get("status", "UNAVAILABLE")
                    ),
                    "advisory_action": lifecycle.get(
                        "advisory_action",
                        private.get("action", "REVIEW"),
                    ),
                    "reason_codes": lifecycle.get(
                        "reason_codes", private.get("reason_codes", [])
                    ),
                    "data_status": (
                        "GO"
                        if private_available
                        else "PRIVATE_POSITION_STATE_UNAVAILABLE"
                    ),
                    "financial_values_scope": "LOCAL_READ_ONLY_UI_ONLY",
                    "execution_authority": "NONE",
                }
            )
        missing_private = sum(
            row["data_status"] != "GO" for row in rows
        )
        return {
            "status": (
                "GO"
                if not missing_private
                and broker_state.get("status") == "BROKER_COUNTS_ALIGNED"
                else "DEGRADED_FAIL_CLOSED"
            ),
            "position_count": len(rows),
            "private_state_missing_count": missing_private,
            "positions": rows,
            "financial_values_scope": "LOCAL_READ_ONLY_UI_ONLY",
            "persisted_to_public_artifact": False,
            "strategy_ownership_inferred": False,
            "automatic_execution": False,
            "execution_authority": "NONE",
        }

    def _private_position_management_states(
        self,
    ) -> dict[str, dict[str, Any]]:
        database = (
            self.project_root
            / "data/portfolio/private/position_management.sqlite3"
        )
        if not database.is_file():
            return {}
        try:
            uri = f"file:{database.as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                rows = connection.execute(
                    "SELECT position_identity, payload_json "
                    "FROM position_events ORDER BY observed_at, rowid"
                ).fetchall()
        except sqlite3.Error:
            return {}
        latest = {}
        for identity, raw in rows:
            try:
                payload = json.loads(str(raw))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                latest[str(identity)] = payload
        return latest

    def _verified_private_position_sizing(self) -> dict[str, Any]:
        path = (
            self.project_root
            / "data/portfolio/private/latest-action-plan.json"
        )
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        sizing = payload.get("whole_share_sizing", {})
        reconciliation = self._json(
            PUBLIC_ROOT / "ibkr" / "live" / "reconciliation.json"
        )
        expected_hash = str(
            reconciliation.get("private_snapshot_hash", "")
        )
        if (
            reconciliation.get("status") != "GO"
            or not str(
                reconciliation.get("reconciliation_status", "")
            ).startswith("LIVE_RECONCILED")
            or not expected_hash
            or str(sizing.get("account_snapshot_hash", ""))
            != expected_hash
        ):
            return {}
        return sizing if isinstance(sizing, dict) else {}

    def _build_performance(
        self,
        month: str,
        environment: str,
    ) -> dict[str, Any]:
        year, month_number = (int(part) for part in month.split("-"))
        local_today = datetime.now(ZoneInfo("Europe/Amsterdam")).date()
        records: dict[str, dict[str, Any]] = {}

        history = (
            self.project_root
            / "data"
            / "performance"
            / "private"
            / "daily-pnl.jsonl"
        )
        if history.is_file():
            for line in history.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_environment = str(row.get("environment", "")).upper()
                if not row_environment:
                    source = str(row.get("source", "")).upper()
                    row_environment = (
                        "PAPER" if "PAPER" in source else "LIVE"
                        if "LIVE" in source
                        else "UNKNOWN"
                    )
                if row_environment != environment:
                    continue
                key = str(row.get("session_date", ""))
                if key:
                    records[key] = {
                        "realized_pnl_eur": _finite(row.get("realized_pnl_eur")),
                        "unrealized_pnl_eur": _finite(row.get("unrealized_pnl_eur")),
                        "net_pnl_eur": _finite(row.get("net_pnl_eur")) or 0.0,
                        "source": row.get("source"),
                        "evidence": row.get("evidence"),
                    }

        if environment == "PAPER":
            paper = self._json(
                PUBLIC_ROOT
                / "ibkr"
                / "phase9"
                / "position-ledger-audit.json"
            )
            projection = paper.get("partial_close_projection", {})
            projection_day = _date_key(projection.get("last_updated_at"))
            realized = _finite(projection.get("realized_pnl_eur"))
            if projection_day and realized is not None:
                records[projection_day] = {
                    "realized_pnl_eur": realized,
                    "unrealized_pnl_eur": None,
                    "net_pnl_eur": realized,
                    "source": "PAPER_LEDGER_AUDIT",
                    "evidence": "HISTORICAL_PAPER_PROJECTION",
                }

        live = self._json(PUBLIC_ROOT / "ibkr" / "live" / "performance.json")
        if environment == "LIVE":
            live_day = _date_key(live.get("generated_at"))
            live_realized = _finite(live.get("realized_pnl_eur"))
            live_unrealized = _finite(live.get("unrealized_pnl_eur"))
            if live_day and (
                live_realized is not None or live_unrealized is not None
            ):
                records[live_day] = {
                    "realized_pnl_eur": live_realized,
                    "unrealized_pnl_eur": live_unrealized,
                    "net_pnl_eur": (
                        (live_realized or 0.0)
                        + (live_unrealized or 0.0)
                    ),
                    "source": "RECONCILED_LIVE_PERFORMANCE",
                    "evidence": "BROKER_DERIVED",
                }

        target = self._json(PUBLIC_ROOT / "capital" / "daily_profit_target.json")
        weeks = []
        observed_values = []
        for week in calendar.Calendar(firstweekday=0).monthdatescalendar(year, month_number):
            cells = []
            for day in week:
                key = day.isoformat()
                record = records.get(key)
                if day.month != month_number:
                    status = "OUTSIDE_MONTH"
                elif day > local_today:
                    status = "FUTURE"
                elif record:
                    status = "OBSERVED"
                    observed_values.append(float(record["net_pnl_eur"]))
                elif day.weekday() >= 5:
                    status = "NON_TRADING_DAY"
                else:
                    status = "UNAVAILABLE"
                cells.append(
                    {
                        "date": key,
                        "day": day.day,
                        "status": status,
                        "record": record,
                        "is_today": day == local_today,
                    }
                )
            weeks.append(cells)

        previous_month = f"{year - 1}-12" if month_number == 1 else f"{year:04d}-{month_number - 1:02d}"
        next_month = f"{year + 1}-01" if month_number == 12 else f"{year:04d}-{month_number + 1:02d}"
        period_performance = _period_performance(
            records,
            today=local_today,
            current_equity_eur=(
                self._latest_private_equity_eur()
                if environment == "LIVE"
                else None
            ),
        )
        return self._sanitize(
            {
                "schema": "ui_pnl_calendar_v1",
                "status": "GO",
                "month": month,
                "environment": environment,
                "available_environments": ["LIVE", "PAPER"],
                "environment_separation": "STRICT_NO_MIXING",
                "month_label": f"{calendar.month_name[month_number]} {year}",
                "previous_month": previous_month,
                "next_month": next_month,
                "weeks": weeks,
                "summary": {
                    "observed_days": len(observed_values),
                    "winning_days": sum(value > 0 for value in observed_values),
                    "losing_days": sum(value < 0 for value in observed_values),
                    "net_pnl_eur": sum(observed_values),
                    "best_day_eur": max(observed_values) if observed_values else None,
                    "worst_day_eur": min(observed_values) if observed_values else None,
                },
                "daily_target": {
                    "session_date": target.get("session_date"),
                    "target_eur": target.get("daily_profit_target_eur"),
                    "reported_pnl_eur": target.get("net_daily_pnl_eur"),
                    "input_source": target.get("input_source", "UNAVAILABLE"),
                    "counted_as_broker_pnl": False,
                },
                "period_performance": period_performance,
                "live_performance_status": live.get("status", "UNAVAILABLE"),
                "execution_authority": "NONE",
                "broker_calls": 0,
            }
        )

    def _latest_private_equity_eur(self) -> float | None:
        return _finite(
            self._latest_private_account_metrics().get(
                "portfolio_value_eur"
            )
        )

    def _latest_private_account_metrics(self) -> dict[str, Any]:
        reconciliation = self._json(
            PUBLIC_ROOT / "ibkr" / "live" / "reconciliation.json"
        )
        expected_hash = str(
            reconciliation.get("private_snapshot_hash", "")
        )
        live_status = self._json(
            PUBLIC_ROOT / "ibkr" / "live" / "status.json"
        )
        reconciliation_state = str(
            reconciliation.get("reconciliation_status")
            or live_status.get("account_reconciliation")
            or ""
        )
        if (
            reconciliation.get("status") != "GO"
            or not reconciliation_state.startswith("LIVE_RECONCILED")
        ):
            return _private_account_unavailable(
                "LIVE_RECONCILIATION_NOT_PROVEN"
            )
        database = (
            self.project_root
            / "data/execution/live/private/broker_observation.sqlite3"
        )
        if not expected_hash or not database.is_file():
            return _private_account_unavailable(
                "LIVE_RECONCILIATION_HASH_UNAVAILABLE"
            )
        try:
            uri = f"file:{database.as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                rows = connection.execute(
                    "SELECT snapshot_hash, payload_json, created_at "
                    "FROM snapshots ORDER BY created_at"
                ).fetchall()
        except sqlite3.Error:
            return _private_account_unavailable(
                "PRIVATE_ACCOUNT_STORE_READ_FAILED"
            )
        if not rows or str(rows[-1][0]) != expected_hash:
            return _private_account_unavailable(
                "PRIVATE_SNAPSHOT_HASH_MISMATCH"
            )

        history: list[tuple[str, float]] = []
        latest_values: dict[str, float] = {}
        latest_completed_at = None
        for _snapshot_hash, payload_json, created_at in rows:
            try:
                payload = json.loads(str(payload_json))
            except json.JSONDecodeError:
                continue
            account = payload.get("account", {})
            if account.get("status") not in {None, "COMPLETE"}:
                continue
            values: dict[str, float] = {}
            for item in account.get("values", []):
                if str(item.get("currency", "")).upper() != "EUR":
                    continue
                value = _finite(item.get("value"))
                if value is not None:
                    values[str(item.get("tag", ""))] = value
            equity = values.get("NetLiquidation")
            if equity is None or equity <= 0:
                continue
            completed_at = str(
                payload.get("snapshot_completed_at") or created_at
            )
            history.append((completed_at, equity))
            if str(_snapshot_hash) == expected_hash:
                latest_values = values
                latest_completed_at = completed_at

        equity = latest_values.get("NetLiquidation")
        if equity is None or equity <= 0:
            return _private_account_unavailable(
                "VERIFIED_NET_LIQUIDATION_EUR_UNAVAILABLE"
            )
        high_water = max(value for _, value in history)
        drawdown = max(0.0, 1.0 - equity / high_water)
        cash = latest_values.get("TotalCashValue")
        gross_position_value = latest_values.get("GrossPositionValue")
        return {
            "status": "GO",
            "source": "HASH_VERIFIED_PRIVATE_LIVE_OBSERVATION",
            "observed_at": latest_completed_at,
            "portfolio_value_eur": round(equity, 2),
            "portfolio_value_status": "GO",
            "cash_eur": round(cash, 2) if cash is not None else None,
            "cash_status": "GO" if cash is not None else "UNAVAILABLE",
            "gross_position_value_eur": (
                round(gross_position_value, 2)
                if gross_position_value is not None
                else None
            ),
            "high_water_mark_eur": round(high_water, 2),
            "high_water_mark_status": (
                "GO" if len(history) >= 2 else "SINGLE_OBSERVATION"
            ),
            "current_drawdown_pct": round(drawdown, 8),
            "observation_count": len(history),
            "financial_values_scope": "LOCAL_READ_ONLY_UI_ONLY",
            "persisted_to_public_artifact": False,
        }

    def _build_news(self) -> dict[str, Any]:
        payload = self._json(NEWS_DIGEST)
        group_coverage = self._json(GROUP_COVERAGE)
        sector_analysis = self._json(SECTOR_ANALYSIS)
        industry_analysis = self._json(INDUSTRY_ANALYSIS)
        ibkr_news = self._json(IBKR_NEWS_CAPABILITIES)
        event_intelligence = self._json(NEWS_INTELLIGENCE_STATUS)
        material_events = self._json(NEWS_MATERIAL_EVENTS)
        portfolio_impact = self._json(NEWS_PORTFOLIO_IMPACT)
        event_study = self._json(NEWS_EVENT_STUDY_STATUS)
        rows = []
        for row in payload.get("important_news", []):
            title = str(row.get("title", "")).strip()
            symbols = [
                str(symbol).upper()
                for symbol in row.get("symbols", [])
            ]
            rows.append(
                {
                    "title": title,
                    "source": row.get("source"),
                    "published_at": row.get("published_at"),
                    "importance": row.get("importance"),
                    "importance_score": row.get("importance_score"),
                    "direction": row.get("direction"),
                    "sentiment_polarity": row.get(
                        "sentiment_polarity"
                    ),
                    "symbols": symbols,
                    "external_search_url": (
                        "https://news.google.com/search?q="
                        + quote_plus(
                            f"{title} {' '.join(symbols[:3])}"
                        )
                    ),
                    "link_semantics": (
                        "SEARCH_LINK_NOT_ARCHIVED_SOURCE_URL"
                    ),
                }
            )
        macro_events = []
        for row in payload.get("upcoming_macro_events", []):
            macro_events.append(
                {
                    "name": row.get("name"),
                    "scheduled_at": row.get("scheduled_at"),
                    "importance": row.get("importance"),
                    "affected_markets": row.get(
                        "affected_markets", []
                    ),
                    "source_url": row.get("source_url"),
                    "schedule_source": row.get("schedule_source"),
                }
            )
        return self._sanitize(
            {
                "schema": "ui_market_news_v1",
                "status": payload.get("status", "NO_DATA"),
                "generated_at": payload.get("generated_at"),
                "freshness_status": payload.get(
                    "news_freshness_status", "UNAVAILABLE"
                ),
                "event_risk_within_24h": payload.get(
                    "event_risk_within_24h", False
                ),
                "important_news": rows,
                "important_news_count": len(rows),
                "upcoming_macro_events": macro_events,
                "source_status": payload.get(
                    "news_source_status", {}
                ),
                "sector_industry_coverage": {
                    "status": group_coverage.get("status", "NO_DATA"),
                    "sector_count": group_coverage.get("sector_count", 0),
                    "industry_count": group_coverage.get(
                        "industry_count", 0
                    ),
                    "fundamental_coverage_ratio": group_coverage.get(
                        "signal_eligible_fundamental_coverage_ratio"
                    ),
                    "fundamental_missing_symbols": group_coverage.get(
                        "signal_eligible_fundamental_missing_symbols", []
                    ),
                },
                "top_sectors": sector_analysis.get("groups", [])[:8],
                "top_industries": industry_analysis.get("groups", [])[:8],
                "ibkr_news": {
                    "status": ibkr_news.get("status", "NOT_PROBED"),
                    "provider_count": ibkr_news.get("provider_count", 0),
                    "historical_headlines_capability": ibkr_news.get(
                        "historical_headlines_capability", "UNPROVEN"
                    ),
                    "tws_connected": ibkr_news.get("tws_connected", False),
                },
                "event_intelligence": {
                    "status": event_intelligence.get(
                        "status", "NOT_RUN"
                    ),
                    "raw_article_count": event_intelligence.get(
                        "raw_article_count", 0
                    ),
                    "deduplicated_story_count": event_intelligence.get(
                        "deduplicated_story_count", 0
                    ),
                    "material_event_count": event_intelligence.get(
                        "material_event_count", 0
                    ),
                    "mapped_symbol_count": event_intelligence.get(
                        "mapped_symbol_count", 0
                    ),
                    "event_class_count": event_intelligence.get(
                        "event_class_count", 0
                    ),
                    "portfolio_impact_event_count": portfolio_impact.get(
                        "portfolio_impact_event_count", 0
                    ),
                    "event_classifier": event_intelligence.get(
                        "event_classifier", "NOT_RUN"
                    ),
                    "calibration": event_intelligence.get(
                        "calibration", "NOT_RUN"
                    ),
                    "standalone_entry_allowed": False,
                    "execution_authority": "NONE",
                },
                "material_events": [
                    {
                        "story_cluster_id": row.get("story_cluster_id"),
                        "title": row.get("title"),
                        "published_at": row.get("last_published_at"),
                        "event_classes": row.get("event_classes", []),
                        "symbols": row.get("symbols", [])[:8],
                        "sentiment_score": row.get("sentiment_score"),
                        "materiality": row.get("materiality"),
                        "independent_source_count": row.get(
                            "independent_source_count", 0
                        ),
                    }
                    for row in material_events.get("rows", [])[:10]
                ],
                "event_study": {
                    "status": event_study.get("status", "NOT_RUN"),
                    "complete_label_count": event_study.get(
                        "complete_label_count", 0
                    ),
                    "causal_label_count": event_study.get(
                        "causal_training_eligible_label_count", 0
                    ),
                    "descriptive_label_count": event_study.get(
                        "historical_descriptive_complete_label_count", 0
                    ),
                    "pending_label_count": event_study.get(
                        "pending_or_unavailable_label_count", 0
                    ),
                    "horizons": event_study.get("horizon_summary", []),
                    "model_readiness": event_study.get(
                        "model_readiness",
                        {
                            "status": "NOT_RUN",
                            "smoke_ready": False,
                            "serious_shadow_ready": False,
                        },
                    ),
                    "published_at_training_eligible": False,
                    "execution_authority": "NONE",
                },
                "news_is_context_only": True,
                "automatic_execution": False,
                "execution_authority": "NONE",
            }
        )

    def _build_research(self) -> dict[str, Any]:
        registry = self._json(STRATEGY_REGISTRY)
        phase = self._json(
            PUBLIC_ROOT / "research" / "phase11_14" / "status.json"
        )
        leaderboard = self._json(
            PUBLIC_ROOT / "research" / "autopilot" / "leaderboard.json"
        )
        role_leaderboards = self._json(
            PUBLIC_ROOT
            / "research"
            / "role_leaderboards"
            / "status.json"
        )
        active_swing = self._json(
            PUBLIC_ROOT / "research" / "active_swing" / "status.json"
        )
        rl_status = self._json(PUBLIC_ROOT / "rl" / "status.json")
        p4_readiness = self._json(
            PUBLIC_ROOT / "verification" / "p4-readiness.json"
        )
        return self._sanitize(
            {
                "schema": "ui_research_viewmodel_v1",
                "status": "GO" if registry else "NO_DATA",
                "strategy_registry": {
                    "bulk_strategy_count": registry.get(
                        "bulk_strategy_count", 0
                    ),
                    "standard_strategy_count": registry.get(
                        "standard_strategy_count", 0
                    ),
                    "generated_at": registry.get("generated_at"),
                },
                "phase11_14": phase,
                "leaderboard": _leaderboard_summary(leaderboard),
                "functional_leaderboards": role_leaderboards,
                "active_swing_sprints_3_6": active_swing,
                "rl": rl_status,
                "p4": p4_readiness,
                "future_holdout": "COLLECTING_NOT_INDEPENDENT_YET",
                "execution_authority": "NONE",
                "broker_calls": 0,
            }
        )

    def _build_health(self) -> dict[str, Any]:
        sources = data_source_status(self.project_root)
        return self._sanitize(
            {
                "schema": "ui_health_viewmodel_v1",
                "status": "GO",
                "runtime": self._json(Path("runtime/heartbeat.json")),
                "machine": self._json(
                    PUBLIC_ROOT
                    / "operations"
                    / "machine-status.json"
                ),
                "execution": self._json(
                    PUBLIC_ROOT
                    / "operations"
                    / "execution-status.json"
                ),
                "live": self._json(
                    PUBLIC_ROOT / "ibkr" / "live" / "status.json"
                ),
                "phase9": self._json(
                    PUBLIC_ROOT / "ibkr" / "phase9" / "status.json"
                ),
                "telegram": self._telegram_status(),
                "data_sources": sources,
                "ui_read_only": True,
            }
        )

    def _build_audit(self) -> dict[str, Any]:
        files = (
            PUBLIC_ROOT / "operations" / "last-cycle.json",
            PUBLIC_ROOT / "operations" / "signal-lifecycle.json",
            PUBLIC_ROOT / "portfolio" / "lifecycle-audit.json",
            PUBLIC_ROOT / "portfolio" / "sizing-audit.json",
            PUBLIC_ROOT / "ibkr" / "phase9" / "audit.json",
        )
        records = []
        for path in files:
            payload = self._json(path)
            if payload:
                records.append(
                    {
                        "artifact": str(path).replace("\\", "/"),
                        "modified_at": datetime.fromtimestamp(
                            (self.project_root / path).stat().st_mtime,
                            UTC,
                        ).isoformat(),
                        "payload": payload,
                    }
                )
        return self._sanitize(
            {
                "schema": "ui_audit_viewmodel_v1",
                "status": "GO",
                "records": records,
                "secrets_exposed": False,
                "raw_account_ids_exposed": False,
                "ui_mutations_enabled": False,
            }
        )

    def _telegram_status(self) -> dict[str, Any]:
        candidates = (
            PUBLIC_ROOT / "notifications" / "telegram_status.json",
            PUBLIC_ROOT / "notifications" / "telegram-status.json",
            PUBLIC_ROOT / "notifications" / "status.json",
        )
        for path in candidates:
            payload = self._json(path)
            if payload:
                return payload
        return {"status": "UNAVAILABLE"}

    def _json(self, relative: Path) -> dict[str, Any]:
        path = self.project_root / relative
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _frame(self, relative: Path) -> pd.DataFrame:
        path = self.project_root / relative
        if not path.is_file():
            return pd.DataFrame()
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return pd.DataFrame()
        key = str(relative)
        with self._lock:
            cached = self._frames.get(key)
            if cached and cached[0] == mtime:
                return cached[1]
        for attempt in range(3):
            try:
                frame = (
                    pd.read_csv(path)
                    if path.suffix.casefold() == ".csv"
                    else pd.read_parquet(path)
                )
            except (OSError, ValueError):
                if attempt < 2:
                    time.sleep(0.05)
                    continue
                with self._lock:
                    stale = self._frames.get(key)
                return stale[1] if stale else pd.DataFrame()
            with self._lock:
                self._frames[key] = (mtime, frame)
            return frame
        return pd.DataFrame()

    def _cached(
        self,
        key: str,
        builder: Any,
        *,
        ttl: float | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic()
        ttl = self.ttl_seconds if ttl is None else ttl
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] <= ttl:
                return cached[1]
        value = builder()
        with self._lock:
            self._cache[key] = (now, value)
        return value

    @staticmethod
    def _values(frame: pd.DataFrame, column: str) -> list[str]:
        return sorted(
            {
                str(value)
                for value in frame[column].dropna().unique()
                if str(value).strip()
            }
        )

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._sanitize(item)
                for key, item in value.items()
                if not FORBIDDEN_KEYS.search(str(key))
            }
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize(item) for item in value]
        if isinstance(value, Path):
            return value.name
        if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value


def _private_account_unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": reason,
        "source": "HASH_VERIFIED_PRIVATE_LIVE_OBSERVATION",
        "observed_at": None,
        "portfolio_value_eur": None,
        "portfolio_value_status": reason,
        "cash_eur": None,
        "cash_status": reason,
        "gross_position_value_eur": None,
        "high_water_mark_eur": None,
        "high_water_mark_status": reason,
        "current_drawdown_pct": None,
        "observation_count": 0,
        "financial_values_scope": "LOCAL_READ_ONLY_UI_ONLY",
        "persisted_to_public_artifact": False,
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _metric_raw(
    metrics: dict[str, Any],
    metric: str,
) -> float | None:
    row = metrics.get(metric, {})
    return _finite(row.get("raw")) if isinstance(row, dict) else None


def _rolling_strategy_performance(
    frame: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    required = {"strategy_id", "fold_id", "cost_bps", "date", "daily_return"}
    if frame.empty or not required.issubset(frame.columns):
        return {}
    data = frame.loc[:, sorted(required)].copy()
    data["cost_bps"] = pd.to_numeric(data["cost_bps"], errors="coerce")
    data["daily_return"] = pd.to_numeric(
        data["daily_return"], errors="coerce"
    )
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    data = data.loc[
        data["cost_bps"].eq(10.0)
        & data["date"].notna()
        & data["daily_return"].notna()
    ]
    windows = {
        "1h": 252,
        "2h": 126,
        "4h": 126,
        "1d": 63,
        "1w": 26,
        "1mo": 12,
    }
    annualization = {
        "1h": 1638.0,
        "2h": 819.0,
        "4h": 410.0,
        "1d": 252.0,
        "1w": 52.0,
        "1mo": 12.0,
    }
    freshness_limits_days = {
        "1h": 2.0,
        "2h": 3.0,
        "4h": 7.0,
        "1d": 10.0,
        "1w": 21.0,
        "1mo": 45.0,
    }
    result: dict[str, dict[str, Any]] = {}
    for strategy_id, group in data.groupby("strategy_id", sort=False):
        ordered = group.sort_values("date")
        duplicate_count = int(ordered.duplicated(["date"]).sum())
        if duplicate_count:
            result[str(strategy_id)] = {
                "status": "BLOCKED_DUPLICATE_OOS_TIMESTAMPS",
                "window_observations": 0,
                "available_observations": len(ordered),
                "duplicate_observations": duplicate_count,
                "return": None,
                "period_profit_factor": None,
                "annualized_sharpe": None,
                "maximum_drawdown": None,
                "last_observation": None,
                "cost_bps": 10.0,
                "freshness_status": "BLOCKED_DUPLICATE_OOS_TIMESTAMPS",
                "age_days": None,
            }
            continue
        fold_id = str(ordered.iloc[-1]["fold_id"])
        timeframe = fold_id.split("_", maxsplit=1)[0]
        window = windows.get(timeframe, 63)
        recent = ordered.tail(window)
        returns = recent["daily_return"].astype(float)
        cumulative = (1.0 + returns).cumprod()
        peak = cumulative.cummax()
        drawdowns = cumulative.div(peak).sub(1.0)
        positive = float(returns.loc[returns > 0].sum())
        negative = abs(float(returns.loc[returns < 0].sum()))
        standard_deviation = float(returns.std(ddof=1))
        complete = len(recent) >= window
        last_observation = recent.iloc[-1]["date"]
        age_days = max(
            0.0,
            (
                datetime.now(UTC) - last_observation.to_pydatetime()
            ).total_seconds()
            / 86400.0,
        )
        freshness_status = (
            "CURRENT_OOS_WINDOW"
            if age_days <= freshness_limits_days.get(timeframe, 10.0)
            else "STALE_HISTORICAL_OOS_WINDOW"
        )
        result[str(strategy_id)] = {
            "status": "GO" if complete else "INSUFFICIENT_ROLLING_WINDOW",
            "timeframe": timeframe,
            "window_observations": window,
            "available_observations": len(ordered),
            "used_observations": len(recent),
            "duplicate_observations": 0,
            "return": float(cumulative.iloc[-1] - 1.0),
            "period_profit_factor": (
                positive / negative if negative > 0 else None
            ),
            "profit_factor_status": (
                "GO"
                if negative > 0
                else "PERFECT_NO_NEGATIVE_PERIODS"
                if positive > 0
                else "NO_NONZERO_PERIOD_RETURNS"
            ),
            "annualized_sharpe": (
                float(
                    returns.mean()
                    / standard_deviation
                    * annualization.get(timeframe, 252.0) ** 0.5
                )
                if standard_deviation > 0
                else None
            ),
            "maximum_drawdown": float(drawdowns.min()),
            "last_observation": last_observation.isoformat(),
            "cost_bps": 10.0,
            "metric_scope": "OOS_PERIOD_RETURNS",
            "freshness_status": freshness_status,
            "age_days": age_days,
        }
    return result


def _signal_classification(score: float) -> str:
    if score >= 0.85:
        return "A+"
    if score >= 0.78:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.62:
        return "C"
    return "REJECT"


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _date_key(value: Any) -> str | None:
    parsed = _timestamp(value)
    return parsed.astimezone(ZoneInfo("Europe/Amsterdam")).date().isoformat() if parsed else None


def _period_performance(
    records: dict[str, dict[str, Any]],
    *,
    today: date,
    current_equity_eur: float | None,
) -> list[dict[str, Any]]:
    dated: list[tuple[date, float]] = []
    for key, row in records.items():
        try:
            session_date = date.fromisoformat(key)
        except ValueError:
            continue
        value = _finite(row.get("net_pnl_eur"))
        if value is not None and session_date <= today:
            dated.append((session_date, value))
    dated.sort(key=lambda item: item[0])
    starts = (
        ("TODAY", today),
        ("WEEK_TO_DATE", today - timedelta(days=today.weekday())),
        ("MONTH_TO_DATE", today.replace(day=1)),
        ("YEAR_TO_DATE", today.replace(month=1, day=1)),
    )
    rows = []
    for period, start in starts:
        values = [value for session, value in dated if start <= session <= today]
        net_pnl = sum(values)
        implied_start_equity = (
            current_equity_eur - net_pnl
            if current_equity_eur is not None
            else None
        )
        return_pct = (
            net_pnl / implied_start_equity
            if implied_start_equity is not None
            and implied_start_equity > 0
            and values
            else None
        )
        rows.append(
            {
                "period": period,
                "start_date": start.isoformat(),
                "end_date": today.isoformat(),
                "status": "GO" if values else "NO_OBSERVATIONS",
                "observed_days": len(values),
                "winning_days": sum(value > 0 for value in values),
                "losing_days": sum(value < 0 for value in values),
                "net_pnl_eur": round(net_pnl, 2) if values else None,
                "return_pct": (
                    round(return_pct, 8) if return_pct is not None else None
                ),
                "return_method": (
                    "RECONCILED_PNL_OVER_IMPLIED_START_EQUITY"
                    if return_pct is not None
                    else "EQUITY_BASE_OR_PNL_HISTORY_UNAVAILABLE"
                ),
                "cash_flow_adjusted": False,
                "exact_equity_public": False,
            }
        )
    return rows


def _validated_month(value: str | None) -> str:
    if value:
        try:
            parsed = datetime.strptime(value, "%Y-%m")
            if 2000 <= parsed.year <= 2100:
                return value
        except ValueError:
            pass
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m")


EXCHANGES = (
    ("NYSE", "XNYS", "America/New_York", 40.71, -74.01, "United States"),
    ("NASDAQ", "XNAS", "America/New_York", 40.75, -73.99, "United States"),
    ("LSE", "XLON", "Europe/London", 51.51, -0.09, "United Kingdom"),
    ("EURONEXT AMS", "XAMS", "Europe/Amsterdam", 52.37, 4.90, "Netherlands"),
    ("EURONEXT PAR", "XPAR", "Europe/Paris", 48.86, 2.35, "France"),
    ("XETRA", "XETR", "Europe/Berlin", 50.11, 8.68, "Germany"),
    ("TSE", "XTKS", "Asia/Tokyo", 35.68, 139.77, "Japan"),
    ("HKEX", "XHKG", "Asia/Hong_Kong", 22.28, 114.16, "Hong Kong"),
    ("SSE", "XSHG", "Asia/Shanghai", 31.23, 121.47, "China"),
    ("BSE", "XBOM", "Asia/Kolkata", 19.08, 72.88, "India"),
    ("ASX", "XASX", "Australia/Sydney", -33.87, 151.21, "Australia"),
    ("TSX", "XTSE", "America/Toronto", 43.65, -79.38, "Canada"),
)


def _exchange_clock(now: datetime | None = None) -> list[dict[str, Any]]:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    minute = pd.Timestamp(current).floor("min")
    rows = []
    for name, calendar_name, timezone, latitude, longitude, country in EXCHANGES:
        status = "CALENDAR_UNAVAILABLE"
        next_event = None
        next_event_type = None
        try:
            exchange = xcals.get_calendar(calendar_name)
            is_open = bool(exchange.is_open_on_minute(minute))
            status = "OPEN" if is_open else "CLOSED"
            if is_open:
                next_event = exchange.next_close(minute)
                next_event_type = "CLOSES"
            else:
                next_event = exchange.next_open(minute)
                next_event_type = "OPENS"
        except (KeyError, ValueError, IndexError):
            pass
        next_event_local = (
            next_event.tz_convert(timezone).isoformat()
            if next_event is not None
            else None
        )
        seconds_to_next_event = (
            max(
                0,
                int(
                    (
                        next_event.to_pydatetime().astimezone(UTC)
                        - current
                    ).total_seconds()
                ),
            )
            if next_event is not None
            else None
        )
        rows.append(
            {
                "name": name,
                "calendar": calendar_name,
                "timezone": timezone,
                "country": country,
                "latitude": latitude,
                "longitude": longitude,
                "status": status,
                "local_time": current.astimezone(ZoneInfo(timezone)).isoformat(),
                "next_event_type": next_event_type,
                "next_event_utc": next_event.isoformat() if next_event is not None else None,
                "next_event_local": next_event_local,
                "seconds_to_next_event": seconds_to_next_event,
            }
        )
    return rows


def _leaderboard_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("actual_market_rows", [])
    source = "ACTUAL_MARKET"
    if not rows:
        rows = payload.get("technical_fixture_rows", [])
        source = "TECHNICAL_FIXTURE_NOT_FINANCIAL_EVIDENCE"
    compact = []
    for row in rows[:20]:
        metrics = row.get("metrics", {})
        compact.append(
            {
                "strategy_id": row.get("strategy_id"),
                "stage": row.get("stage"),
                "status": row.get("status"),
                "CAGR": metrics.get("CAGR"),
                "Sharpe": metrics.get("Sharpe"),
                "period_profit_factor": metrics.get(
                    "period_profit_factor"
                ),
                "maximum_drawdown": metrics.get("maximum_drawdown"),
                "trade_episodes": metrics.get("trade_episodes"),
            }
        )
    return {
        "status": payload.get("status", "NO_DATA"),
        "generated_at": payload.get("generated_at"),
        "source": source,
        "actual_market_count": payload.get("actual_market_count", 0),
        "technical_fixture_count": payload.get(
            "technical_fixture_count", 0
        ),
        "fixture_results_are_financial_evidence": False,
        "rows": compact,
    }
